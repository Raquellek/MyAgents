"""
message_strategy_dqn_gru.py 

python3 message_strategy_dqn_gru.py --msg-dim 32 --k-neighbors 6 --red-strategy swarming 
============================
Combines:
  • GRU-DQN (CNN → LayerNorm → 2-layer GRU → Dueling Q-heads)  [strategy_dqn_gru.py]
  • Learned inter-agent message passing (k-NN mean-pool)         [message_module.py / model.py]

Architecture per agent
----------------------
  obs (H×W×C)  ──CNN──►  features
                               │
  agg_msg (msg_dim) ──────────cat──► Linear → LayerNorm → ReLU → hidden_dim
                                                                       │
                                                               2-layer GRU
                                                                  │      │
                                                            Dueling Q   msg_encoder
                                                           (value+adv)   │
                                                                        new_msg → neighbours next step

Training design
---------------
  • Parameter sharing per team (one network shared by all red / all blue agents)
  • Double DQN + soft target updates
  • Experience replay storing (state, msg, action, reward, next_state, next_msg, done) sequences
  • ε-greedy exploration decayed per episode
  • Both teams trained simultaneously (RED offensive vs BLUE defensive)
  • Env recreated each episode to avoid magent2 C++ backend segfaults
"""

from magent2.environments import battle_v4
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

from message_module import MessageAggregator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Experience = namedtuple(
    'Experience',
    ['state', 'msg', 'action', 'reward', 'next_state', 'next_msg', 'done']
)


# ─────────────────────────────────────────────────────────────────────────────
# Position extraction from parallel_env
# ─────────────────────────────────────────────────────────────────────────────

def get_team_positions(env, team_prefix: str, active_agents) -> dict:
    """
    Extract (x, y) grid positions for alive team agents.
    Falls back to zeros if the C++ backend doesn't expose positions.
    """
    positions = {}
    try:
        inner   = env.unwrapped
        handles = inner.get_handles()
        t_idx   = 0 if team_prefix.startswith("red") else 1
        if t_idx < len(handles):
            pos   = inner.get_pos(handles[t_idx])
            names = [a for a in active_agents if a.startswith(team_prefix)]
            for i, name in enumerate(names):
                if i < len(pos):
                    positions[name] = (int(pos[i][0]), int(pos[i][1]))
    except Exception:
        for a in active_agents:
            if a.startswith(team_prefix):
                positions[a] = (0, 0)
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Replay Memory — stores (state, msg, …) sequences
# ─────────────────────────────────────────────────────────────────────────────

class ReplayMemory:
    """Circular buffer; samples contiguous sequences that don't cross episodes."""

    def __init__(self, capacity: int, sequence_length: int):
        self.capacity        = capacity
        self.memory: list    = []
        self.position        = 0
        self.sequence_length = sequence_length

    def push(self, state, msg, action, reward, next_state, next_msg, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(
            state, msg, action, reward, next_state, next_msg, done
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        n = len(self.memory)
        if n < self.sequence_length:
            return None
        indices = random.sample(
            range(n - self.sequence_length + 1),
            min(batch_size, n - self.sequence_length + 1),
        )
        sequences = []
        for start in indices:
            # Reject sequences that cross episode boundaries
            if all(not self.memory[i].done
                   for i in range(start, start + self.sequence_length - 1)):
                sequences.append(
                    [self.memory[i] for i in range(start, start + self.sequence_length)]
                )
        return sequences if sequences else None

    def __len__(self):
        return len(self.memory)


# ─────────────────────────────────────────────────────────────────────────────
# Network: CNN → concat(msg) → LayerNorm → GRU → Dueling Q + msg_encoder
# ─────────────────────────────────────────────────────────────────────────────

class MessageGRUDQNNetwork(nn.Module):
    """
    GRU-DQN augmented with inter-agent message passing.

    Inputs per forward call
    -----------------------
    state_sequence : (batch, seq_len, H, W, C)  – spatial observations
    msg_sequence   : (batch, seq_len, msg_dim)   – aggregated neighbour msgs

    Outputs
    -------
    q_values : (batch, action_dim)  – Dueling Q-values at last timestep
    new_msg  : (batch, msg_dim)     – message to broadcast to neighbours
    """

    def __init__(self, input_shape: tuple, hidden_dim: int, output_dim: int,
                 sequence_length: int, msg_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_shape     = input_shape
        self.hidden_dim      = hidden_dim
        self.output_dim      = output_dim
        self.sequence_length = sequence_length
        self.msg_dim         = msg_dim
        self.num_layers      = num_layers

        # ── CNN feature extractor ────────────────────────────────────────
        if len(input_shape) == 3:
            H, W, C = input_shape
            self.use_cnn = True
            self.cnn = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            )
            cnn_out_dim = 64 * H * W
        else:
            self.use_cnn = False
            cnn_out_dim  = int(np.prod(input_shape))

        # ── Fusion: CNN features + aggregated message ────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + msg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── Temporal context ─────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.1 if num_layers > 1 else 0.0,
        )

        # ── Dueling Q-heads ──────────────────────────────────────────────
        half = hidden_dim // 2
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, half), nn.ReLU(), nn.Linear(half, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, half), nn.ReLU(), nn.Linear(half, output_dim)
        )

        # ── Message encoder head ─────────────────────────────────────────
        self.msg_encoder = nn.Sequential(
            nn.Linear(hidden_dim, half), nn.ReLU(),
            nn.Linear(half, msg_dim),
            nn.Tanh(),   # bounded in [-1, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.advantage_stream[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.msg_encoder[-2].weight, gain=0.01)

    def _extract_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, H, W, C) → (N, cnn_out_dim)"""
        if self.use_cnn:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.cnn(x)
        return x.reshape(x.size(0), -1)

    def forward(self, state_sequence: torch.Tensor,
                msg_sequence: torch.Tensor):
        """
        state_sequence : (batch, seq, *input_shape)
        msg_sequence   : (batch, seq, msg_dim)
        """
        batch, seq = state_sequence.shape[:2]

        # Extract CNN features for every (batch, time) pair
        flat_states = state_sequence.reshape(batch * seq, *self.input_shape)
        cnn_feats   = self._extract_cnn(flat_states)                      # (B*T, cnn_dim)
        flat_msgs   = msg_sequence.reshape(batch * seq, self.msg_dim)      # (B*T, msg_dim)

        # Fuse spatial features with communication context
        fused    = self.fusion(torch.cat([cnn_feats, flat_msgs], dim=-1))  # (B*T, hidden)
        fused    = fused.reshape(batch, seq, self.hidden_dim)

        # Temporal reasoning
        gru_out, _ = self.gru(fused)                                       # (B, T, hidden)
        h          = gru_out[:, -1, :]                                     # last timestep

        # Dueling Q
        value     = self.value_stream(h)                                   # (B, 1)
        advantage = self.advantage_stream(h)                               # (B, action_dim)
        q_values  = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Outgoing message
        new_msg = self.msg_encoder(h)                                      # (B, msg_dim)

        return q_values, new_msg


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class MessageStrategicDQNAgent:
    """
    GRU-DQN agent with learned inter-agent message passing.

    Each agent maintains:
      - A per-agent state buffer  (deque of observations, length = seq_len)
      - A per-agent message buffer (deque of aggregated msgs, length = seq_len)
    Both are passed to select_action_for_agent() and stored in replay.
    """

    def __init__(
        self,
        input_shape:    tuple,
        action_dim:     int,
        strategy_name:  str,
        msg_dim:        int   = 16,
        k_neighbors:    int   = 4,
        hidden_dim:     int   = 256,
        sequence_length: int  = 4,
        learning_rate:  float = 0.0003,
        gamma:          float = 0.99,
        tau:            float = 0.01,
        memory_size:    int   = 50_000,
        batch_size:     int   = 128,
        epsilon_start:  float = 1.0,
        epsilon_end:    float = 0.05,
        epsilon_decay:  float = 0.998,
    ):
        self.input_shape     = input_shape
        self.action_dim      = action_dim
        self.strategy_name   = strategy_name
        self.msg_dim         = msg_dim
        self.hidden_dim      = hidden_dim
        self.sequence_length = sequence_length
        self.gamma           = gamma
        self.tau             = tau
        self.batch_size      = batch_size
        self.epsilon         = epsilon_start
        self.epsilon_end     = epsilon_end
        self.epsilon_decay   = epsilon_decay

        self.policy_net = MessageGRUDQNNetwork(
            input_shape, hidden_dim, action_dim, sequence_length, msg_dim
        ).to(device)
        self.target_net = MessageGRUDQNNetwork(
            input_shape, hidden_dim, action_dim, sequence_length, msg_dim
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler  = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)
        self.memory     = ReplayMemory(memory_size, sequence_length)
        self.aggregator = MessageAggregator(msg_dim=msg_dim, k=k_neighbors)

        self._setup_strategy()
        self.episode_rewards    = []
        self.episode_losses     = []
        self.evaluation_rewards = []

    # ── Strategy reward shaping ──────────────────────────────────────────────

    def _setup_strategy(self):
        presets = {
            "offensive": (2.0, 0.5, 0.8),
            "defensive": (0.7, 2.0, 1.5),
            "flanking":  (1.5, 0.8, 0.5),
            "swarming":  (1.2, 0.6, 2.0),
        }
        a, d, g = presets.get(self.strategy_name, (1.0, 1.0, 1.0))
        self.attack_reward_multiplier   = a
        self.defense_reward_multiplier  = d
        self.grouping_reward_multiplier = g

    def _shape_reward(self, reward: float, info: dict) -> float:
        shaped = reward
        if info.get('damage_dealt', 0) > 0:
            shaped += 0.5 * self.attack_reward_multiplier
        return shaped

    # ── Message aggregation ──────────────────────────────────────────────────

    def aggregate_messages(self, agent_ids: list, positions: dict,
                           msg_state: dict) -> dict:
        """Delegate to the MessageAggregator for this team."""
        return self.aggregator.aggregate(agent_ids, positions, msg_state)

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action_for_agent(self, state: np.ndarray, agg_msg: np.ndarray,
                                 state_buffer: deque, msg_buffer: deque,
                                 training: bool = True) -> tuple:
        """
        Select action and produce outgoing message.

        Parameters
        ----------
        state       : current observation
        agg_msg     : aggregated message from neighbours
        state_buffer: per-agent deque of past obs (len = sequence_length)
        msg_buffer  : per-agent deque of past agg_msgs
        training    : if True, applies ε-greedy

        Returns
        -------
        (action: int, new_msg: np.ndarray)
        """
        state_buffer.append(state)
        msg_buffer.append(agg_msg)

        if training and random.random() < self.epsilon:
            action  = random.randrange(self.action_dim)
            new_msg = np.zeros(self.msg_dim, dtype=np.float32)
            return action, new_msg

        if len(state_buffer) < self.sequence_length:
            action  = random.randrange(self.action_dim)
            new_msg = np.zeros(self.msg_dim, dtype=np.float32)
            return action, new_msg

        with torch.no_grad():
            s_seq = torch.FloatTensor(
                np.array(list(state_buffer), dtype=np.float32)
            ).unsqueeze(0).to(device)                        # (1, seq, *shape)
            m_seq = torch.FloatTensor(
                np.array(list(msg_buffer), dtype=np.float32)
            ).unsqueeze(0).to(device)                        # (1, seq, msg_dim)
            q_vals, new_msg_t = self.policy_net(s_seq, m_seq)

        action  = q_vals.max(1)[1].item()
        new_msg = new_msg_t.squeeze(0).cpu().numpy()
        return action, new_msg

    # ── Memory ───────────────────────────────────────────────────────────────

    def store_transition(self, state, msg, action, reward,
                         next_state, next_msg, done, info=None):
        shaped = self._shape_reward(reward, info or {})
        self.memory.push(state, msg, action, shaped, next_state, next_msg, done)

    # ── Learning ─────────────────────────────────────────────────────────────

    def learn(self):
        if len(self.memory) < self.batch_size * self.sequence_length:
            return None
        sequences = self.memory.sample(self.batch_size)
        if sequences is None or len(sequences) < 2:
            return None

        B   = len(sequences)
        SL  = self.sequence_length
        IS  = self.input_shape

        states      = np.zeros((B, SL) + IS, dtype=np.float32)
        next_states = np.zeros_like(states)
        msgs        = np.zeros((B, SL, self.msg_dim), dtype=np.float32)
        next_msgs   = np.zeros_like(msgs)
        actions     = np.zeros(B, dtype=np.int64)
        rewards     = np.zeros(B, dtype=np.float32)
        dones       = np.zeros(B, dtype=np.float32)

        for i, seq in enumerate(sequences):
            for j, exp in enumerate(seq):
                states[i, j]      = exp.state
                next_states[i, j] = exp.next_state
                msgs[i, j]        = exp.msg
                next_msgs[i, j]   = exp.next_msg
            last = seq[-1]
            actions[i] = last.action
            rewards[i] = last.reward
            dones[i]   = last.done

        s  = torch.FloatTensor(states).to(device)
        ns = torch.FloatTensor(next_states).to(device)
        m  = torch.FloatTensor(msgs).to(device)
        nm = torch.FloatTensor(next_msgs).to(device)
        a  = torch.LongTensor(actions).unsqueeze(1).to(device)
        r  = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        d  = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q, _ = self.policy_net(s, m)
        current_q    = current_q.gather(1, a)

        with torch.no_grad():
            # Double DQN: policy net selects, target net evaluates
            next_q_policy, _ = self.policy_net(ns, nm)
            next_a           = next_q_policy.max(1)[1].unsqueeze(1)
            next_q_target, _ = self.target_net(ns, nm)
            next_q           = next_q_target.gather(1, next_a)

        target_q = r + self.gamma * next_q * (1 - d)
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft target update
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            'policy_net':    self.policy_net.state_dict(),
            'target_net':    self.target_net.state_dict(),
            'optimizer':     self.optimizer.state_dict(),
            'epsilon':       self.epsilon,
            'strategy_name': self.strategy_name,
            'msg_dim':       self.msg_dim,
            'metrics': {
                'episode_rewards':    self.episode_rewards,
                'episode_losses':     self.episode_losses,
                'evaluation_rewards': self.evaluation_rewards,
            },
        }, path)
        with open(f"{path}_params.json", 'w') as f:
            json.dump({
                'strategy_name':            self.strategy_name,
                'msg_dim':                  self.msg_dim,
                'attack_reward_multiplier': self.attack_reward_multiplier,
                'defense_reward_multiplier': self.defense_reward_multiplier,
                'grouping_reward_multiplier': self.grouping_reward_multiplier,
            }, f, indent=4)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt.get('epsilon', self.epsilon)
        if 'metrics' in ckpt:
            m = ckpt['metrics']
            self.episode_rewards    = m.get('episode_rewards', [])
            self.episode_losses     = m.get('episode_losses', [])
            self.evaluation_rewards = m.get('evaluation_rewards', [])
        return True

    def plot_metrics(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.plot(self.episode_rewards, label='Episode Rewards')
        if self.evaluation_rewards:
            ex = np.linspace(0, len(self.episode_rewards), len(self.evaluation_rewards))
            ax1.plot(ex, self.evaluation_rewards, 'r-', label='Evaluation Rewards')
        ax1.set_title(f'{self.strategy_name} – Rewards')
        ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward')
        ax1.legend(); ax1.grid(True)
        if self.episode_losses:
            ax2.plot(self.episode_losses, label='Training Loss')
            ax2.set_title(f'{self.strategy_name} – Loss')
            ax2.set_xlabel('Episode'); ax2.set_ylabel('Loss')
            ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path); plt.close()
        # else:
        #     plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Agent construction helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_team_agents(env, red_strategy, blue_strategy,
                      sequence_length, hidden_dim, msg_dim, k_neighbors):
    first_red  = next(a for a in env.possible_agents if a.startswith("red_"))
    first_blue = next(a for a in env.possible_agents if a.startswith("blue_"))
    common = dict(
        msg_dim         = msg_dim,
        k_neighbors     = k_neighbors,
        hidden_dim      = hidden_dim,
        sequence_length = sequence_length,
        learning_rate   = 0.0003,
        gamma           = 0.99,
        tau             = 0.01,
        memory_size     = 50_000,
        batch_size      = 128,
        epsilon_start   = 1.0,
        epsilon_end     = 0.05,
        epsilon_decay   = 0.998,
    )
    red_agent  = MessageStrategicDQNAgent(
        env.observation_spaces[first_red].shape,
        env.action_spaces[first_red].n,
        red_strategy, **common,
    )
    blue_agent = MessageStrategicDQNAgent(
        env.observation_spaces[first_blue].shape,
        env.action_spaces[first_blue].n,
        blue_strategy, **common,
    )
    return red_agent, blue_agent


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_multi_agent(
    num_episodes       = 3000,
    max_steps          = 500,
    update_frequency   = 2,
    eval_interval      = 100,
    checkpoint_interval= 500,
    log_interval       = 10,
    red_strategy       = "offensive",
    blue_strategy      = "defensive",
    sequence_length    = 4,
    hidden_dim         = 256,
    msg_dim            = 16,
    k_neighbors        = 4,
):
    # One-shot env to get spaces, then close
    _env = battle_v4.parallel_env(render_mode=None)
    _env.reset()
    red_agent, blue_agent = _make_team_agents(
        _env, red_strategy, blue_strategy,
        sequence_length, hidden_dim, msg_dim, k_neighbors,
    )
    n_red  = sum(1 for a in _env.possible_agents if a.startswith("red_"))
    n_blue = sum(1 for a in _env.possible_agents if a.startswith("blue_"))
    print(f"Training: RED ({red_strategy}, {n_red} agents) vs "
          f"BLUE ({blue_strategy}, {n_blue} agents)")
    print(f"Obs shape: {_env.observation_spaces[_env.possible_agents[0]].shape}  "
          f"Actions: {_env.action_spaces[_env.possible_agents[0]].n}  "
          f"msg_dim: {msg_dim}  k: {k_neighbors}")
    _env.close()

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    red_ckpt  = f"msg_dqn_red_{red_strategy}_{ts}"
    blue_ckpt = f"msg_dqn_blue_{blue_strategy}_{ts}"

    for episode in range(1, num_episodes + 1):
        # Fresh C++ backend each episode to avoid segfaults
        env               = battle_v4.parallel_env(render_mode=None)
        observations, _   = env.reset()

        # Per-agent GRU context buffers
        state_buffers = {a: deque(maxlen=sequence_length) for a in env.possible_agents}
        msg_buffers   = {a: deque(maxlen=sequence_length) for a in env.possible_agents}

        # Per-agent outgoing message state (updated after every step)
        red_msg_state  = {a: np.zeros(msg_dim, dtype=np.float32)
                          for a in env.possible_agents if a.startswith("red")}
        blue_msg_state = {a: np.zeros(msg_dim, dtype=np.float32)
                          for a in env.possible_agents if a.startswith("blue")}

        red_ep_r, blue_ep_r     = 0.0, 0.0
        red_losses, blue_losses = [], []
        step = 0

        while observations and step < max_steps:
            active = list(observations.keys())

            # ── Aggregate messages per team ───────────────────────────────
            red_active  = [a for a in active if a.startswith("red")]
            blue_active = [a for a in active if a.startswith("blue")]

            red_positions  = get_team_positions(env, "red",  active)
            blue_positions = get_team_positions(env, "blue", active)

            red_agg  = red_agent.aggregate_messages(
                red_active,  red_positions,  red_msg_state)
            blue_agg = blue_agent.aggregate_messages(
                blue_active, blue_positions, blue_msg_state)

            # ── Select actions for every active agent ─────────────────────
            actions      = {}
            current_msgs = {}   # agg_msg seen by each agent this step
            new_msgs     = {}   # outgoing message produced this step

            for agent_name, obs in observations.items():
                if agent_name.startswith("red"):
                    agg_msg = red_agg.get(agent_name, np.zeros(msg_dim, np.float32))
                    action, new_msg = red_agent.select_action_for_agent(
                        obs, agg_msg,
                        state_buffers[agent_name],
                        msg_buffers[agent_name],
                    )
                    red_msg_state[agent_name] = new_msg
                else:
                    agg_msg = blue_agg.get(agent_name, np.zeros(msg_dim, np.float32))
                    action, new_msg = blue_agent.select_action_for_agent(
                        obs, agg_msg,
                        state_buffers[agent_name],
                        msg_buffers[agent_name],
                    )
                    blue_msg_state[agent_name] = new_msg

                actions[agent_name]      = action
                current_msgs[agent_name] = agg_msg
                new_msgs[agent_name]     = new_msg

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # ── Aggregate next-step messages for next_msg storage ─────────
            next_active      = list(next_obs.keys())
            next_red_active  = [a for a in next_active if a.startswith("red")]
            next_blue_active = [a for a in next_active if a.startswith("blue")]
            next_red_pos     = get_team_positions(env, "red",  next_active)
            next_blue_pos    = get_team_positions(env, "blue", next_active)
            next_red_agg     = red_agent.aggregate_messages(
                next_red_active,  next_red_pos,  red_msg_state)
            next_blue_agg    = blue_agent.aggregate_messages(
                next_blue_active, next_blue_pos, blue_msg_state)

            # ── Store transitions ─────────────────────────────────────────
            for agent_name, obs in observations.items():
                if agent_name not in rewards:
                    continue
                done   = (terminations.get(agent_name, True) or
                          truncations.get(agent_name, False))
                n_ob   = next_obs.get(agent_name, np.zeros_like(obs))
                info   = infos.get(agent_name, {})

                if agent_name.startswith("red"):
                    next_agg = next_red_agg.get(
                        agent_name, np.zeros(msg_dim, np.float32))
                    red_agent.store_transition(
                        obs,
                        current_msgs[agent_name],
                        actions[agent_name],
                        rewards[agent_name],
                        n_ob,
                        next_agg,
                        done,
                        info,
                    )
                    red_ep_r += rewards[agent_name]
                else:
                    next_agg = next_blue_agg.get(
                        agent_name, np.zeros(msg_dim, np.float32))
                    blue_agent.store_transition(
                        obs,
                        current_msgs[agent_name],
                        actions[agent_name],
                        rewards[agent_name],
                        n_ob,
                        next_agg,
                        done,
                        info,
                    )
                    blue_ep_r += rewards[agent_name]

            # ── Learn periodically ────────────────────────────────────────
            if step % update_frequency == 0:
                rl = red_agent.learn()
                bl = blue_agent.learn()
                if rl is not None: red_losses.append(rl)
                if bl is not None: blue_losses.append(bl)

            observations = next_obs
            step += 1

        # ── End of episode ────────────────────────────────────────────────
        env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        red_agent.decay_epsilon()
        blue_agent.decay_epsilon()
        red_agent.scheduler.step()
        blue_agent.scheduler.step()

        avg_rl = float(np.mean(red_losses))  if red_losses  else 0.0
        avg_bl = float(np.mean(blue_losses)) if blue_losses else 0.0
        red_agent.episode_rewards.append(red_ep_r)
        blue_agent.episode_rewards.append(blue_ep_r)
        red_agent.episode_losses.append(avg_rl)
        blue_agent.episode_losses.append(avg_bl)

        if episode % log_interval == 0:
            print(
                f"Ep {episode:4d}/{num_episodes} | "
                f"RED  r={red_ep_r:8.2f}  loss={avg_rl:.5f}  ε={red_agent.epsilon:.4f} | "
                f"BLUE r={blue_ep_r:8.2f}  loss={avg_bl:.5f}  ε={blue_agent.epsilon:.4f}"
            )

        if episode % checkpoint_interval == 0:
            red_agent.save(f"{red_ckpt}_ep{episode}.pt")
            blue_agent.save(f"{blue_ckpt}_ep{episode}.pt")
            red_agent.plot_metrics(save_path=f"{red_ckpt}_metrics_ep{episode}.png")
            blue_agent.plot_metrics(save_path=f"{blue_ckpt}_metrics_ep{episode}.png")
            print(f"  Checkpoints saved (episode {episode})")

        if episode % eval_interval == 0:
            r_eval, b_eval = evaluate_multi_agent(
                red_agent, blue_agent,
                num_episodes=3,
                sequence_length=sequence_length,
                msg_dim=msg_dim,
            )
            red_agent.evaluation_rewards.append(r_eval)
            blue_agent.evaluation_rewards.append(b_eval)
            print(f"  Eval → RED={r_eval:.2f}  BLUE={b_eval:.2f}")

    red_agent.save(f"{red_ckpt}_final.pt")
    blue_agent.save(f"{blue_ckpt}_final.pt")
    red_agent.plot_metrics(save_path=f"{red_ckpt}_metrics_final.png")
    blue_agent.plot_metrics(save_path=f"{blue_ckpt}_metrics_final.png")
    print("Training complete!")
    return red_agent, blue_agent


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_multi_agent(red_agent, blue_agent, num_episodes=3,
                         sequence_length=4, msg_dim=16):
    """Greedy evaluation. Returns (avg_red_reward, avg_blue_reward)."""
    total_red, total_blue = 0.0, 0.0

    for _ in range(num_episodes):
        env             = battle_v4.parallel_env(render_mode=None)
        observations, _ = env.reset()

        state_buffers  = {a: deque(maxlen=sequence_length) for a in env.possible_agents}
        msg_buffers    = {a: deque(maxlen=sequence_length) for a in env.possible_agents}
        red_msg_state  = {a: np.zeros(msg_dim, np.float32)
                          for a in env.possible_agents if a.startswith("red")}
        blue_msg_state = {a: np.zeros(msg_dim, np.float32)
                          for a in env.possible_agents if a.startswith("blue")}

        red_r = blue_r = 0.0

        while observations:
            active     = list(observations.keys())
            red_active = [a for a in active if a.startswith("red")]
            blue_active= [a for a in active if a.startswith("blue")]
            red_pos    = get_team_positions(env, "red",  active)
            blue_pos   = get_team_positions(env, "blue", active)
            red_agg    = red_agent.aggregate_messages(red_active,  red_pos,  red_msg_state)
            blue_agg   = blue_agent.aggregate_messages(blue_active, blue_pos, blue_msg_state)

            actions = {}
            for agent_name, obs in observations.items():
                if agent_name.startswith("red"):
                    agg_msg = red_agg.get(agent_name, np.zeros(msg_dim, np.float32))
                    action, new_msg = red_agent.select_action_for_agent(
                        obs, agg_msg,
                        state_buffers[agent_name],
                        msg_buffers[agent_name],
                        training=False,
                    )
                    red_msg_state[agent_name] = new_msg
                else:
                    agg_msg = blue_agg.get(agent_name, np.zeros(msg_dim, np.float32))
                    action, new_msg = blue_agent.select_action_for_agent(
                        obs, agg_msg,
                        state_buffers[agent_name],
                        msg_buffers[agent_name],
                        training=False,
                    )
                    blue_msg_state[agent_name] = new_msg
                actions[agent_name] = action

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            for agent_name, r in rewards.items():
                if agent_name.startswith("red"):
                    red_r += r
                else:
                    blue_r += r
            observations = {
                a: o for a, o in next_obs.items()
                if not (terminations.get(a, False) or truncations.get(a, False))
            }

        env.close()
        total_red  += red_r
        total_blue += blue_r

    return total_red / num_episodes, total_blue / num_episodes


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",         type=int,   default=3000)
    p.add_argument("--max-steps",        type=int,   default=500)
    p.add_argument("--red-strategy",     type=str,   default="offensive",
                   choices=["offensive","defensive","flanking","swarming","balanced"])
    p.add_argument("--blue-strategy",    type=str,   default="defensive",
                   choices=["offensive","defensive","flanking","swarming","balanced"])
    p.add_argument("--msg-dim",          type=int,   default=16)
    p.add_argument("--k-neighbors",      type=int,   default=4)
    p.add_argument("--hidden-dim",       type=int,   default=256)
    p.add_argument("--sequence-length",  type=int,   default=4)
    p.add_argument("--log-interval",     type=int,   default=10)
    p.add_argument("--eval-interval",    type=int,   default=100)
    p.add_argument("--checkpoint-interval", type=int, default=10)
    p.add_argument("--update-frequency", type=int,   default=2)
    args = p.parse_args()

    train_multi_agent(
        num_episodes        = args.episodes,
        max_steps           = args.max_steps,
        update_frequency    = args.update_frequency,
        eval_interval       = args.eval_interval,
        checkpoint_interval = args.checkpoint_interval,
        log_interval        = args.log_interval,
        red_strategy        = args.red_strategy,
        blue_strategy       = args.blue_strategy,
        sequence_length     = args.sequence_length,
        hidden_dim          = args.hidden_dim,
        msg_dim             = args.msg_dim,
        k_neighbors         = args.k_neighbors,
    )
