from magent2.environments import battle_v4
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Optional, Tuple, Dict, Any, List
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


# ─────────────────────────────────────────────────────────────────────────────
# Replay Memory
# ─────────────────────────────────────────────────────────────────────────────

class ReplayMemory:
    """Experience replay buffer that samples contiguous sequences."""
    def __init__(self, capacity, sequence_length):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.sequence_length = sequence_length

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.sequence_length:
            return None
        indices = random.sample(
            range(len(self.memory) - self.sequence_length + 1),
            min(batch_size, len(self.memory) - self.sequence_length + 1),
        )
        sequences = []
        for start in indices:
            # Don't cross episode boundaries
            if all(not self.memory[i].done for i in range(start, start + self.sequence_length - 1)):
                sequences.append([self.memory[i] for i in range(start, start + self.sequence_length)])
        return sequences if sequences else None

    def __len__(self):
        return len(self.memory)


# ─────────────────────────────────────────────────────────────────────────────
# Network: CNN → LayerNorm → 2-layer GRU → Dueling Q-heads
# ─────────────────────────────────────────────────────────────────────────────

class GRUDQNNetwork(nn.Module):
    """
    DQN with:
    - CNN feature extractor for spatial (H, W, C) observations
    - LayerNorm for training stability
    - 2-layer GRU for temporal memory
    - Dueling architecture (value + advantage streams) for faster convergence
    """
    def __init__(self, input_shape, hidden_dim, output_dim, sequence_length, num_layers=2):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.num_layers = num_layers

        # Feature extractor: CNN for spatial inputs, MLP otherwise
        if len(input_shape) == 3:
            H, W, C = input_shape
            self.use_cnn = True
            self.cnn = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            cnn_out_dim = 64 * H * W
            self.feature_fc = nn.Sequential(
                nn.Linear(cnn_out_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
        else:
            self.use_cnn = False
            flat_dim = int(np.prod(input_shape))
            self.feature_fc = nn.Sequential(
                nn.Linear(flat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )

        # GRU for temporal context
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Dueling heads: V(s) + A(s, a) − mean(A)
        half = hidden_dim // 2
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, half), nn.ReLU(), nn.Linear(half, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, half), nn.ReLU(), nn.Linear(half, output_dim)
        )

    def _extract_features(self, x):
        """x: (N, *input_shape) → (N, hidden_dim)"""
        if self.use_cnn:
            # (N, H, W, C) → (N, C, H, W)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.cnn(x)
            x = x.reshape(x.size(0), -1)
        else:
            x = x.reshape(x.size(0), -1)
        return self.feature_fc(x)

    def forward(self, state_sequence):
        """state_sequence: (batch, seq_len, *input_shape) → (batch, action_dim)"""
        batch, seq = state_sequence.shape[:2]
        # Flatten batch and time for feature extraction
        flat = state_sequence.reshape(batch * seq, *self.input_shape)
        features = self._extract_features(flat)          # (batch*seq, hidden_dim)
        features = features.reshape(batch, seq, self.hidden_dim)
        gru_out, _ = self.gru(features)                  # (batch, seq, hidden_dim)
        h = gru_out[:, -1, :]                            # last timestep
        value = self.value_stream(h)                     # (batch, 1)
        advantage = self.advantage_stream(h)             # (batch, action_dim)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class StrategicDQNAgent:
    """
    DQN agent using GRU + Dueling + Double DQN.

    Key hyperparameter changes vs. original:
    - learning_rate : 0.0003  (Adam is stable at this rate with CNN)
    - tau           : 0.01    (10× faster target net sync than 0.001)
    - memory_size   : 200 000 (larger buffer absorbs all-agent experiences)
    - batch_size    : 128     (more stable gradient estimates)
    - epsilon_decay : per-episode (call decay_epsilon() once per episode)
    - hidden_dim    : 256     (wider network)
    - num_layers    : 2       (deeper GRU)
    """
    def __init__(
        self,
        input_shape,
        action_dim,
        strategy_name,
        hidden_dim=256,
        sequence_length=4,
        learning_rate=0.0003,
        gamma=0.99,
        tau=0.01,
        memory_size=50_000,
        batch_size=128,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.998,   # applied once per episode
    ):
        self.input_shape = input_shape
        self.action_dim = action_dim
        self.strategy_name = strategy_name
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = GRUDQNNetwork(input_shape, hidden_dim, action_dim, sequence_length).to(device)
        self.target_net = GRUDQNNetwork(input_shape, hidden_dim, action_dim, sequence_length).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # Halve LR every 500 episodes to fine-tune later in training
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)
        self.memory = ReplayMemory(memory_size, sequence_length)
        self.state_buffer = deque(maxlen=sequence_length)  # for single-agent use
        self.total_steps = 0

        self.setup_strategy_rewards()
        self.episode_rewards = []
        self.episode_losses = []
        self.evaluation_rewards = []

    # ── Strategy reward shaping ──────────────────────────────────────────────

    def setup_strategy_rewards(self):
        if self.strategy_name == "offensive":
            self.attack_reward_multiplier   = 2.0
            self.defense_reward_multiplier  = 0.5
            self.grouping_reward_multiplier = 0.8
        elif self.strategy_name == "defensive":
            self.attack_reward_multiplier   = 0.7
            self.defense_reward_multiplier  = 2.0
            self.grouping_reward_multiplier = 1.5
        elif self.strategy_name == "flanking":
            self.attack_reward_multiplier   = 1.5
            self.defense_reward_multiplier  = 0.8
            self.grouping_reward_multiplier = 0.5
        elif self.strategy_name == "swarming":
            self.attack_reward_multiplier   = 1.2
            self.defense_reward_multiplier  = 0.6
            self.grouping_reward_multiplier = 2.0
        else:  # balanced
            self.attack_reward_multiplier   = 1.0
            self.defense_reward_multiplier  = 1.0
            self.grouping_reward_multiplier = 1.0

    def shape_reward(self, reward, state, action, next_state, info):
        shaped = reward
        if info.get('damage_dealt', 0) > 0:
            shaped += 0.5 * self.attack_reward_multiplier
        return shaped

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action_for_agent(self, state, agent_buffer, training=True):
        """
        Select action using a *per-agent* external state buffer.
        Use this in multi-agent training so each agent keeps its own GRU context.
        """
        agent_buffer.append(state)
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        if len(agent_buffer) < self.sequence_length:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            seq = np.array(list(agent_buffer), dtype=np.float32)
            seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, seq, *shape)
            return self.policy_net(seq_t).max(1)[1].item()

    def select_action(self, state, training=True):
        """Single-agent action selection using the internal state buffer."""
        return self.select_action_for_agent(state, self.state_buffer, training)

    # ── Memory ───────────────────────────────────────────────────────────────

    def store_transition(self, state, action, reward, next_state, done, info=None):
        shaped = self.shape_reward(reward, state, action, next_state, info or {})
        self.memory.push(state, action, shaped, next_state, done)
        self.total_steps += 1

    # ── Learning ─────────────────────────────────────────────────────────────

    def learn(self):
        """One gradient update step. Returns loss or None if not ready."""
        if len(self.memory) < self.batch_size * self.sequence_length:
            return None
        sequences = self.memory.sample(self.batch_size)
        if sequences is None or len(sequences) < 2:
            return None

        B = len(sequences)
        states      = np.zeros((B, self.sequence_length) + self.input_shape, dtype=np.float32)
        next_states = np.zeros_like(states)
        actions     = np.zeros(B, dtype=np.int64)
        rewards     = np.zeros(B, dtype=np.float32)
        dones       = np.zeros(B, dtype=np.float32)

        for i, seq in enumerate(sequences):
            for j, exp in enumerate(seq):
                states[i, j]      = exp.state
                next_states[i, j] = exp.next_state
            last = seq[-1]
            actions[i] = last.action
            rewards[i] = last.reward
            dones[i]   = last.done

        s  = torch.FloatTensor(states).to(device)
        ns = torch.FloatTensor(next_states).to(device)
        a  = torch.LongTensor(actions).unsqueeze(1).to(device)
        r  = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        d  = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q = self.policy_net(s).gather(1, a)

        with torch.no_grad():
            # Double DQN: policy net selects action, target net evaluates it
            next_a  = self.policy_net(ns).max(1)[1].unsqueeze(1)
            next_q  = self.target_net(ns).gather(1, next_a)
        target_q = r + self.gamma * next_q * (1 - d)

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft target network update
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)

        return loss.item()

    def decay_epsilon(self):
        """Call once per episode to decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset_state_buffer(self):
        self.state_buffer.clear()

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'epsilon':    self.epsilon,
            'strategy_name': self.strategy_name,
            'metrics': {
                'episode_rewards':    self.episode_rewards,
                'episode_losses':     self.episode_losses,
                'evaluation_rewards': self.evaluation_rewards,
            },
        }, path)
        with open(f"{path}_strategy_params.json", 'w') as f:
            json.dump({
                'strategy_name':            self.strategy_name,
                'attack_reward_multiplier': self.attack_reward_multiplier,
                'defense_reward_multiplier': self.defense_reward_multiplier,
                'grouping_reward_multiplier': self.grouping_reward_multiplier,
            }, f, indent=4)

    def load(self, path):
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(ckpt['policy_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt['epsilon']
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
        else:
            plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent Training  (parameter sharing per team)
# ─────────────────────────────────────────────────────────────────────────────

def _make_team_agents(env, red_strategy, blue_strategy, sequence_length, hidden_dim):
    """Instantiate one shared DQN agent per team."""
    first_red  = next(a for a in env.possible_agents if a.startswith("red_"))
    first_blue = next(a for a in env.possible_agents if a.startswith("blue_"))
    common = dict(
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        learning_rate=0.0003,
        gamma=0.99,
        tau=0.01,
        memory_size=50_000,
        batch_size=128,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.998,   # ~1500 episodes to reach epsilon_end
    )
    red_agent  = StrategicDQNAgent(
        env.observation_spaces[first_red].shape,
        env.action_spaces[first_red].n,
        red_strategy, **common,
    )
    blue_agent = StrategicDQNAgent(
        env.observation_spaces[first_blue].shape,
        env.action_spaces[first_blue].n,
        blue_strategy, **common,
    )
    return red_agent, blue_agent


def train_multi_agent(
    num_episodes=3000,
    max_steps=500,
    update_frequency=2,     # learn every N env steps
    eval_interval=100,
    checkpoint_interval=500,
    log_interval=10,
    red_strategy="offensive",
    blue_strategy="defensive",
    sequence_length=4,
    hidden_dim=256,
):
    """
    Train one shared GRU-DQN policy per team, across ALL red and blue players.

    Parameter sharing means every agent of a team contributes experience to a
    single replay buffer and a single network — sample-efficient and fast.

    Architecture highlights
    -----------------------
    - CNN feature extractor  : captures spatial structure of (13×13×C) obs
    - 2-layer GRU            : short-term temporal memory per agent
    - Dueling Q-heads        : separates state value from action advantage
    - Double DQN             : reduces Q-value overestimation
    - Per-episode ε-decay    : stable exploration schedule regardless of
                               episode length or number of agents
    - LR step scheduler      : halves LR every 500 episodes for fine-tuning

    Hyperparameter rationale
    ------------------------
    lr=0.0003   : Adam sweet-spot for CNNs; avoids instability at 0.001
    tau=0.01    : soft-update 10× faster than original 0.001
    batch=128   : larger minibatch → smoother gradient signal
    memory=200k : absorbs ~12 agents × 500 steps = 6k transitions/episode
    ε-decay/ep  : 0.998^1500 ≈ 0.05 → full exploration for ~1500 episodes
    update_freq=2 : more updates per episode without sacrificing wall-time
    """
    # Create a temporary env just to get space shapes for agent construction
    _env_init = battle_v4.parallel_env(render_mode=None)
    _env_init.reset()

    red_agent, blue_agent = _make_team_agents(
        _env_init, red_strategy, blue_strategy, sequence_length, hidden_dim
    )

    n_red  = sum(1 for a in _env_init.possible_agents if a.startswith("red_"))
    n_blue = sum(1 for a in _env_init.possible_agents if a.startswith("blue_"))
    print(f"Training: RED ({red_strategy}, {n_red} agents) vs BLUE ({blue_strategy}, {n_blue} agents)")
    print(f"Obs shape: {_env_init.observation_spaces[_env_init.possible_agents[0]].shape}  "
          f"Actions: {_env_init.action_spaces[_env_init.possible_agents[0]].n}")
    _env_init.close()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    red_ckpt  = f"dqn_gru_red_{red_strategy}_{ts}"
    blue_ckpt = f"dqn_gru_blue_{blue_strategy}_{ts}"

    for episode in range(1, num_episodes + 1):
        # Recreate env each episode to prevent C++ backend state accumulation
        env = battle_v4.parallel_env(render_mode=None)
        observations, _ = env.reset()

        # Each active agent gets its own GRU context (state buffer)
        state_buffers = {a: deque(maxlen=sequence_length) for a in env.possible_agents}

        red_ep_r, blue_ep_r = 0.0, 0.0
        red_losses, blue_losses = [], []
        step = 0

        while observations and step < max_steps:
            # ── Select actions for every active agent ─────────────────────
            actions = {}
            for agent_name, obs in observations.items():
                team = red_agent if agent_name.startswith("red") else blue_agent
                actions[agent_name] = team.select_action_for_agent(
                    obs, state_buffers[agent_name]
                )

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # ── Store transitions for all agents ──────────────────────────
            for agent_name, obs in observations.items():
                if agent_name not in rewards:
                    continue
                team  = red_agent if agent_name.startswith("red") else blue_agent
                n_ob  = next_obs.get(agent_name, np.zeros_like(obs))
                done  = terminations.get(agent_name, True) or truncations.get(agent_name, False)
                info  = infos.get(agent_name, {})
                team.store_transition(obs, actions[agent_name], rewards[agent_name], n_ob, done, info)

                if agent_name.startswith("red"):
                    red_ep_r += rewards[agent_name]
                else:
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
                num_episodes=3, sequence_length=sequence_length,
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


def evaluate_multi_agent(red_agent, blue_agent, num_episodes=3, sequence_length=4):
    """
    Greedy evaluation of both teams.
    Returns (avg_red_reward, avg_blue_reward).
    """
    env = battle_v4.parallel_env(render_mode=None)
    total_red, total_blue = 0.0, 0.0

    for _ in range(num_episodes):
        observations, _ = env.reset()
        state_buffers = {a: deque(maxlen=sequence_length) for a in env.possible_agents}
        red_r = blue_r = 0.0

        while observations:
            actions = {}
            for agent_name, obs in observations.items():
                team = red_agent if agent_name.startswith("red") else blue_agent
                actions[agent_name] = team.select_action_for_agent(
                    obs, state_buffers[agent_name], training=False
                )
            observations, rewards, terminations, truncations, _ = env.step(actions)
            for agent_name, r in rewards.items():
                if agent_name.startswith("red"):
                    red_r += r
                else:
                    blue_r += r
            # Remove done agents
            observations = {
                a: o for a, o in observations.items()
                if not (terminations.get(a, False) or truncations.get(a, False))
            }

        total_red  += red_r
        total_blue += blue_r

    env.close()
    return total_red / num_episodes, total_blue / num_episodes


# ─────────────────────────────────────────────────────────────────────────────
# Legacy single-agent interface (kept for backwards compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class MAgent2StrategicWrapper(gym.Env):
    """Single-agent Gymnasium wrapper for MAgent2 battle_v4 (legacy)."""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, team_strategy="balanced"):
        super().__init__()
        self.env = battle_v4.parallel_env(render_mode=render_mode)
        self.possible_agents = self.env.possible_agents
        first_agent = self.possible_agents[0]
        self.observation_space = self.env.observation_spaces[first_agent]
        self.action_space      = self.env.action_spaces[first_agent]
        self.render_mode       = render_mode
        self.current_observations = None
        self.agent_name        = first_agent
        self.team_strategy     = team_strategy
        self.agents_info       = {}
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")
        print(f"Team strategy: {self.team_strategy}")

    def reset(self, *, seed=None, options=None):
        result = self.env.reset(seed=seed)
        # PettingZoo parallel_env.reset() returns (obs_dict, info_dict)
        self.current_observations = result[0] if isinstance(result, tuple) else result
        self.agents_info  = {a: {"damage_dealt": 0} for a in self.current_observations}
        self.possible_agents = list(self.current_observations.keys())
        self.agent_name   = self.possible_agents[0]
        return self.current_observations[self.agent_name], {}

    def step(self, action):
        actions = {self.agent_name: action}
        for agent in self.current_observations:
            if agent != self.agent_name:
                actions[agent] = self.env.action_spaces[agent].sample()
        prev_health = self._get_health()
        self.current_observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self._update_agent_info(prev_health, self._get_health())
        if self.agent_name not in self.current_observations:
            if self.current_observations:
                self.agent_name = list(self.current_observations.keys())[0]
            else:
                return np.zeros(self.observation_space.shape), 0.0, True, False, {}
        obs       = self.current_observations[self.agent_name]
        reward    = rewards[self.agent_name]
        terminated = terminations[self.agent_name]
        truncated  = truncations[self.agent_name]
        info = dict(infos.get(self.agent_name, {}))
        info.update(self.agents_info.get(self.agent_name, {}))
        return obs, reward, terminated, truncated, info

    def _get_health(self):
        return {a: {"health": float(np.mean(o))} for a, o in self.current_observations.items()}

    def _update_agent_info(self, prev, curr):
        for a in self.agents_info:
            if a in prev and a in curr:
                diff = prev[a]["health"] - curr[a]["health"]
                if diff < 0:
                    self.agents_info[a]["damage_dealt"] += abs(diff)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def train_strategic_agent(
    env, agent,
    num_episodes=5000, max_steps=1000,
    update_frequency=4, eval_interval=100,
    checkpoint_interval=500, log_interval=10,
):
    """Legacy single-agent training (trains only one agent; prefer train_multi_agent)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt = f"dqn_gru_{agent.strategy_name}_{ts}"
    print(f"Starting training for {agent.strategy_name} strategy, {num_episodes} episodes...")
    episode_rewards, losses = [], []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        agent.reset_state_buffer()
        ep_reward = ep_loss = updates = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, terminated, info)

            if step % update_frequency == 0 and agent.total_steps > agent.sequence_length * agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    ep_loss += loss
                    updates += 1

            ep_reward += reward
            state = next_state
            if terminated or truncated:
                break

        agent.decay_epsilon()
        avg_loss = ep_loss / max(1, updates)
        episode_rewards.append(ep_reward)
        losses.append(avg_loss)
        agent.episode_rewards.append(ep_reward)
        agent.episode_losses.append(avg_loss)

        if episode % log_interval == 0:
            print(f"Ep {episode}/{num_episodes}  reward={ep_reward:.2f}  "
                  f"loss={avg_loss:.6f}  ε={agent.epsilon:.4f}")

        if episode % checkpoint_interval == 0:
            agent.save(f"{ckpt}_ep{episode}.pt")
            agent.plot_metrics(save_path=f"{ckpt}_metrics_ep{episode}.png")

        if episode % eval_interval == 0:
            eval_r = evaluate_strategic_agent(env, agent, num_episodes=5)
            agent.evaluation_rewards.append(eval_r)
            print(f"  Eval at ep {episode}: avg_reward={eval_r:.2f}")

    agent.save(f"{ckpt}_final.pt")
    agent.plot_metrics(save_path=f"{ckpt}_metrics_final.png")
    print("Training complete!")
    return episode_rewards, losses


def evaluate_strategic_agent(env, agent, num_episodes=5):
    total = 0.0
    for _ in range(num_episodes):
        state, _ = env.reset()
        agent.reset_state_buffer()
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    return total / num_episodes


def compare_strategies(strategies, num_episodes=20, render_mode=None):
    """Compare trained single-agent strategies head-to-head."""
    results = {s: 0.0 for s in strategies}
    for _ in range(num_episodes):
        env = MAgent2StrategicWrapper(render_mode=render_mode)
        agents = {}
        for strategy in strategies:
            path = f"dqn_gru_{strategy}_final.pt"
            if os.path.exists(path):
                a = StrategicDQNAgent(
                    env.observation_space.shape,
                    env.action_space.n,
                    strategy, hidden_dim=256, sequence_length=4,
                )
                if a.load(path):
                    agents[strategy] = a
        if not agents:
            continue
        state, _ = env.reset()
        done = False
        strat = random.choice(list(agents.keys()))
        agent = agents[strat]
        agent.reset_state_buffer()
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        results[strat] += reward
    for s in results:
        results[s] /= num_episodes
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Multi-agent training: all red + blue players train simultaneously ────
    #
    # One shared policy per team (parameter sharing).
    # Every player's experience feeds into their team's replay buffer,
    # giving 12–24× more data per episode than single-agent training.
    #
    # Swap strategies as desired:
    #   "offensive" | "defensive" | "flanking" | "swarming" | "balanced"
    red_agent, blue_agent = train_multi_agent(
        num_episodes=3000,
        max_steps=500,
        update_frequency=2,
        eval_interval=100,
        checkpoint_interval=500,
        log_interval=10,
        red_strategy="offensive",
        blue_strategy="defensive",
        sequence_length=4,
        hidden_dim=256,
    )

    # ── Visual playback after training ──────────────────────────────────────
    print("\nRunning visual test...")
    test_env = battle_v4.parallel_env(render_mode="human")
    observations, _ = test_env.reset()
    state_buffers = {a: deque(maxlen=4) for a in test_env.possible_agents}

    while observations:
        actions = {}
        for agent_name, obs in observations.items():
            team = red_agent if agent_name.startswith("red") else blue_agent
            actions[agent_name] = team.select_action_for_agent(
                obs, state_buffers[agent_name], training=False
            )
        observations, _, terminations, truncations, _ = test_env.step(actions)
        observations = {
            a: o for a, o in observations.items()
            if not (terminations.get(a, False) or truncations.get(a, False))
        }

    test_env.close()
