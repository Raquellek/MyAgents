from magent2.environments import battle_v4
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

Experience = namedtuple(
    "Experience",
    ["state", "action", "reward", "next_state", "done"]
)


class ReplayMemory:
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(
            Experience(state, action, reward, next_state, done)
        )

    def sample_sequences(self, batch_size, sequence_length):
        if len(self.memory) < sequence_length + batch_size:
            return None

        sequences = []

        for _ in range(batch_size):
            start = random.randint(0, len(self.memory) - sequence_length)
            seq = list(self.memory)[start:start + sequence_length]

            valid = True
            for exp in seq[:-1]:
                if exp.done:
                    valid = False
                    break

            if valid:
                sequences.append(seq)

        if len(sequences) == 0:
            return None

        return sequences

    def __len__(self):
        return len(self.memory)


class GRUDQN(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=128):
        super().__init__()

        self.obs_shape = obs_shape
        self.input_dim = int(np.prod(obs_shape))

        self.feature = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        batch, seq = x.shape[0], x.shape[1]

        x = x.reshape(batch, seq, -1)
        x = self.feature(x)

        out, _ = self.gru(x)

        last = out[:, -1, :]
        q = self.head(last)

        return q


class DQNGRUAgent:
    def __init__(
        self,
        obs_shape,
        action_dim,
        strategy="offensive",
        sequence_length=4,
        hidden_dim=128,
        lr=1e-4,
        gamma=0.99,
        batch_size=32,
        memory_size=50000
    ):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.strategy = strategy
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_net = GRUDQN(obs_shape, action_dim, hidden_dim).to(device)
        self.target_net = GRUDQN(obs_shape, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.agent_buffers = {}

    def get_buffer(self, agent_name):
        if agent_name not in self.agent_buffers:
            self.agent_buffers[agent_name] = deque(maxlen=self.sequence_length)
        return self.agent_buffers[agent_name]

    def reset_buffers(self):
        self.agent_buffers = {}

    def select_action(self, agent_name, obs, training=True):
        buffer = self.get_buffer(agent_name)
        buffer.append(obs)

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        if len(buffer) < self.sequence_length:
            return random.randrange(self.action_dim)

        state_seq = np.array(buffer, dtype=np.float32)
        state_seq = torch.tensor(state_seq).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.policy_net(state_seq)

        return int(torch.argmax(q_values, dim=1).item())

    def shape_reward(self, reward, agent_name, done):
        shaped = reward

        if self.strategy == "offensive":
            shaped += reward * 1.5

        elif self.strategy == "defensive":
            if not done:
                shaped += 0.02
            if reward < 0:
                shaped += reward * 0.5

        elif self.strategy == "survivor":
            if not done:
                shaped += 0.05
            if done:
                shaped -= 1.0

        elif self.strategy == "swarming":
            if not done:
                shaped += 0.03
            shaped += reward * 1.2

        elif self.strategy == "balanced":
            shaped = reward

        return shaped

    def learn(self):
        sequences = self.memory.sample_sequences(
            self.batch_size,
            self.sequence_length
        )

        if sequences is None:
            return None

        batch_size = len(sequences)

        states = np.zeros(
            (batch_size, self.sequence_length) + self.obs_shape,
            dtype=np.float32
        )
        next_states = np.zeros(
            (batch_size, self.sequence_length) + self.obs_shape,
            dtype=np.float32
        )
        actions = np.zeros(batch_size, dtype=np.int64)
        rewards = np.zeros(batch_size, dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)

        for i, seq in enumerate(sequences):
            for j, exp in enumerate(seq):
                states[i, j] = exp.state
                next_states[i, j] = exp.next_state

            last = seq[-1]
            actions[i] = last.action
            rewards[i] = last.reward
            dones[i] = last.done

        states = torch.tensor(states).to(device)
        next_states = torch.tensor(next_states).to(device)
        actions = torch.tensor(actions).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(device)
        dones = torch.tensor(dones).unsqueeze(1).to(device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "strategy": self.strategy,
            "obs_shape": self.obs_shape,
            "action_dim": self.action_dim,
            "sequence_length": self.sequence_length
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.epsilon = ckpt["epsilon"]


def train():
    strategy = "offensive"

    env = battle_v4.parallel_env(
        render_mode=None,
        map_size=45,
        max_cycles=500
    )

    obs = env.reset(seed=42)

    first_agent = env.agents[0]
    obs_shape = obs[first_agent].shape
    action_dim = env.action_space(first_agent).n

    print("Observation shape:", obs_shape)
    print("Action dim:", action_dim)

    agent = DQNGRUAgent(
        obs_shape=obs_shape,
        action_dim=action_dim,
        strategy=strategy,
        sequence_length=4,
        hidden_dim=128,
        lr=1e-4,
        batch_size=32
    )

    num_episodes = 2000
    update_every = 4
    target_update_every = 20
    save_every = 100

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        agent.reset_buffers()

        episode_reward = 0
        step_count = 0
        losses = []

        while env.agents:
            actions = {}

            for name in env.agents:
                if name.startswith("red"):
                    actions[name] = agent.select_action(
                        name,
                        obs[name],
                        training=True
                    )
                else:
                    actions[name] = env.action_space(name).sample()

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            for name in actions:
                if name.startswith("red"):
                    done = terminations.get(name, False) or truncations.get(name, False)

                    if name in obs:
                        state = obs[name]
                    else:
                        continue

                    if name in next_obs:
                        next_state = next_obs[name]
                    else:
                        next_state = np.zeros(obs_shape, dtype=np.float32)

                    reward = rewards.get(name, 0.0)
                    shaped_reward = agent.shape_reward(reward, name, done)

                    agent.memory.push(
                        state,
                        actions[name],
                        shaped_reward,
                        next_state,
                        done
                    )

                    episode_reward += reward

            obs = next_obs
            step_count += 1

            if step_count % update_every == 0 and len(agent.memory) > 1000:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

            if step_count % target_update_every == 0:
                agent.update_target()

        agent.epsilon = max(
            agent.epsilon_min,
            agent.epsilon * agent.epsilon_decay
        )

        avg_loss = np.mean(losses) if losses else 0

        print(
            f"Episode {episode}/{num_episodes} | "
            f"Reward: {episode_reward:.2f} | "
            f"Loss: {avg_loss:.5f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

        if episode % save_every == 0:
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/dqn_gru_{strategy}_ep{episode}.pt"
            agent.save(path)
            print("Saved:", path)

    agent.save(f"checkpoints/dqn_gru_{strategy}_final.pt")
    print("Training finished")


if __name__ == "__main__":
    train()