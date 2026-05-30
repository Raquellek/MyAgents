# lab3.ipynb

Source notebook: [`lab3.ipynb`](../lab3.ipynb)

## Cell 1

```python
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output
from pathlib import Path

import random, os.path, math, glob, csv, base64, itertools, sys
import gym
from gym.wrappers.record_video import RecordVideo
from pprint import pprint

# The following code is will be used to visualize the environments.

def show_video(directory):
    html = []
    for mp4 in Path(directory).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
    
display = Display(visible=0, size=(1400, 900))
display.start()

def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
```

## Cell 2

```python
# We load CartPole-v1
env = gym.make('CartPole-v1')
# We wrap it in order to save our experiment on a file.
env = RecordVideo(env, './video',  episode_trigger = lambda episode_number: True)
```

## Cell 3

```python
done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
env.close()
```

## Cell 4

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim_observation, n_actions):
        super(Model, self).__init__()

        self.n_actions = n_actions
        self.dim_observation = dim_observation

        self.net = nn.Sequential(
            nn.Linear(self.dim_observation, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, self.n_actions),
            nn.Softmax(dim=-1)
        )

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.saved_log_probs.append(dist.log_prob(action))

        return action.item()
```

## Cell 5

```python
env_id = 'CartPole-v1'
env = gym.make(env_id)
model = Model(env.observation_space.shape[0], env.action_space.n)
print(f'The model we created correspond to:\n{model}')
```

## Cell 6

```python
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import gym

from gym.wrappers import RecordVideo


class BaseAgent:
    
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config['env_id'])
        make_seed(config['seed'])

        try:
            self.env.reset(seed=config['seed'])
        except:
            self.env.seed(config['seed'])

        self.model = Model(
            self.env.observation_space.shape[0],
            self.env.action_space.n
        )

        self.gamma = config['gamma']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )

        self.monitor_env = RecordVideo(
            gym.make(config['env_id']),   
            './video',
            episode_trigger=lambda episode_number: True
        )
    
    def _make_returns(self, rewards):
        returns = np.zeros(len(rewards), dtype=np.float32)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + self.gamma * returns[t + 1]
        return returns
    
    def optimize_model(self, n_trajectories):
        trajectory_rewards = []
        policy_losses = []

        for _ in range(n_trajectories):
            out = self.env.reset()
            if isinstance(out, tuple):
                observation, _ = out
            else:
                observation = out

            done = False
            rewards = []
            log_probs = []

            while not done:
                observation = torch.tensor(observation, dtype=torch.float32)

                probs = self.model(observation)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                step_out = self.env.step(action.item())

                if len(step_out) == 5:
                    next_observation, reward, terminated, truncated, info = step_out
                    done = terminated or truncated
                else:
                    next_observation, reward, done, info = step_out

                rewards.append(reward)
                log_probs.append(log_prob)
                observation = next_observation

            returns = self._make_returns(rewards)
            returns = torch.tensor(returns, dtype=torch.float32)

            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            for log_prob, G in zip(log_probs, returns):
                policy_losses.append(-log_prob * G)

            trajectory_rewards.append(np.sum(rewards))

        policy_loss = torch.stack(policy_losses).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return np.array(trajectory_rewards)
    
    def train(self, n_trajectories, n_update):
        rewards = []
        for episode in range(n_update):
            rewards.append(self.optimize_model(n_trajectories))
            print(
                f'Episode {episode + 1}/{n_update}: '
                f'rewards {round(rewards[-1].mean(), 2)} +/- {round(rewards[-1].std(), 2)}'
            )
        
        r = pd.DataFrame(
            itertools.chain(*(itertools.product([i], rewards[i]) for i in range(len(rewards)))),
            columns=['Epoch', 'Reward']
        )
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd')
        
    def evaluate(self):
        out = self.monitor_env.reset()
        if isinstance(out, tuple):
            observation, _ = out
        else:
            observation = out

        reward_episode = 0
        done = False
            
        while not done:
            observation = torch.tensor(observation, dtype=torch.float32)

            with torch.no_grad():
                probs = self.model(observation)
                action = torch.argmax(probs).item()

            step_out = self.monitor_env.step(action)

            if len(step_out) == 5:
                observation, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                observation, reward, done, info = step_out

            reward_episode += reward

        print(f'Reward: {reward_episode}')
        self.monitor_env.close()
```

## Cell 7

```python
class SimpleAgent(BaseAgent):
    
    def optimize_model(self, n_trajectories):
        
        all_policy_losses = []
        reward_trajectories = []

        for _ in range(n_trajectories):

            out = self.env.reset()
            if isinstance(out, tuple):
                state, _ = out
            else:
                state = out

            done = False
            rewards = []
            log_probs = []

            # === нэг trajectory цуглуулах ===
            while not done:
                state = torch.tensor(state, dtype=torch.float32)

                probs = self.model(state)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                log_prob = dist.log_prob(action)

                step_out = self.env.step(action.item())

                if len(step_out) == 5:
                    next_state, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_out

                rewards.append(reward)
                log_probs.append(log_prob)

                state = next_state

            # === discounted return ===
            returns = self._make_returns(rewards)
            returns = torch.tensor(returns, dtype=torch.float32)

            # normalize (optional but important)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # === loss ===
            for log_prob, G in zip(log_probs, returns):
                all_policy_losses.append(-log_prob * G)

            reward_trajectories.append(sum(rewards))

        # === final loss ===
        loss = torch.stack(all_policy_losses).sum()

        # gradient step (already given)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return np.array(reward_trajectories)
```

## Cell 8

```python
#@title Config {display-mode: "form", run: "auto"}

env_id = 'CartPole-v1'  #@param ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
learning_rate = 0.01  #@param {type: "number"}
gamma = 1  #@param {type: "number"}
seed = 1235  #@param {type: "integer"}
#@markdown ---

config = {
    'env_id': env_id,
    'learning_rate': learning_rate,
    'seed': seed,
    'gamma': gamma
}

print("Current config is:")
pprint(config)
```

## Cell 9

```python
agent = SimpleAgent(config)
agent.train(n_trajectories=50, n_update=50)
```

## Cell 10

```python
agent.evaluate()
```

## Cell 11

```python
class EnhancedAgent(BaseAgent):
    
    def optimize_model(self, n_trajectories):
        reward_trajectories = []
        all_losses = []

        for _ in range(n_trajectories):
            out = self.env.reset()
            if isinstance(out, tuple):
                state, _ = out
            else:
                state = out

            done = False
            rewards = []
            log_probs = []

            # collect one trajectory
            while not done:
                state = torch.tensor(state, dtype=torch.float32)

                probs = self.model(state)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                step_out = self.env.step(action.item())

                if len(step_out) == 5:
                    next_state, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_out

                rewards.append(reward)
                log_probs.append(log_prob)
                state = next_state

            # future discounted returns G_t
            returns = self._make_returns(rewards)
            returns = torch.tensor(returns, dtype=torch.float32)

            # optional normalization
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # enhanced REINFORCE loss: each action gets its own future return
            for log_prob, G_t in zip(log_probs, returns):
                all_losses.append(-log_prob * G_t)

            reward_trajectories.append(sum(rewards))

        loss = torch.stack(all_losses).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return np.array(reward_trajectories)
```

## Cell 12

```python
#@title Config Enhanced {display-mode: "form", run: "auto"}

env_id = 'CartPole-v1'  #@param ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
learning_rate = 0.001  #@param {type: "number"}
gamma = 1  #@param {type: "number"}
seed = 1  #@param {type: "integer"}
#@markdown ---

config_enhanced = {
    'env_id': env_id,
    'learning_rate': learning_rate,
    'seed': seed,
    'gamma': gamma
}

print("Current config_enhanced is:")
pprint(config_enhanced)
```

## Cell 13

```python
agent = EnhancedAgent(config_enhanced)
agent.train(n_trajectories=50, n_update=50)
```

## Cell 14

```python
agent.evaluate()
```

## Cell 15

```python
class ActorNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=-1)
        return out
    
    def select_action(self, x):
        return torch.multinomial(self(x), 1).detach().numpy()
```

## Cell 16

```python
class ValueNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
```

## Cell 17

```python
class A2CAgent:

    def __init__(self, config):
        self.config = config
        self.env = gym.make(config['env_id'])
        make_seed(config['seed'])

        try:
            self.env.reset(seed=config['seed'])
        except:
            self.env.seed(config['seed'])

        self.gamma = config['gamma']

        # video env тусдаа байх нь зөв
        self.monitor_env = gym.make(config['env_id'])

        self.value_network = ValueNetwork(self.env.observation_space.shape[0], 16, 1)
        self.actor_network = ActorNetwork(self.env.observation_space.shape[0], 16, self.env.action_space.n)

        self.value_network_optimizer = optim.RMSprop(
            self.value_network.parameters(),
            lr=config['value_network']['learning_rate']
        )
        self.actor_network_optimizer = optim.RMSprop(
            self.actor_network.parameters(),
            lr=config['actor_network']['learning_rate']
        )

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), [next_value], axis=0)

        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def training_batch(self, epochs, batch_size):
        episode_count = 0
        actions = np.empty((batch_size,), dtype=np.int64)
        dones = np.empty((batch_size,), dtype=np.bool_)
        rewards = np.empty((batch_size,), dtype=np.float32)
        values = np.empty((batch_size,), dtype=np.float32)
        observations = np.empty((batch_size,) + self.env.observation_space.shape, dtype=np.float32)

        out = self.env.reset()
        observation = out[0] if isinstance(out, tuple) else out

        rewards_test = []

        for epoch in range(epochs):
            for i in range(batch_size):
                observations[i] = observation

                obs_t = torch.tensor(observation, dtype=torch.float32)
                values[i] = self.value_network(obs_t).detach().item()

                policy = self.actor_network(obs_t)
                action = torch.multinomial(policy, 1).item()
                actions[i] = action

                step_out = self.env.step(action)
                if len(step_out) == 5:
                    observation, rewards[i], terminated, truncated, _ = step_out
                    dones[i] = terminated or truncated
                else:
                    observation, rewards[i], dones[i], _ = step_out

                if dones[i]:
                    out = self.env.reset()
                    observation = out[0] if isinstance(out, tuple) else out

            if dones[-1]:
                next_value = 0.0
            else:
                next_value = self.value_network(
                    torch.tensor(observation, dtype=torch.float32)
                ).detach().item()

            episode_count += dones.sum()

            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
            self.optimize_model(observations, actions, returns, advantages)

            if epoch % 50 == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate() for _ in range(20)]))
                print(
                    f"Epoch {epoch}/{epochs}: Mean rewards: "
                    f"{round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}"
                )

                if rewards_test[-1].mean() > 490 and epoch != epochs - 1:
                    print("Early stopping!")
                    break

                out = self.env.reset()
                observation = out[0] if isinstance(out, tuple) else out

    def optimize_model(self, observations, actions, returns, advantages):
        observations = torch.tensor(observations, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        returns = torch.tensor(returns[:, None], dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # Critic
        values = self.value_network(observations)              # [B, 1]
        value_loss = F.mse_loss(values, returns)

        self.value_network_optimizer.zero_grad()
        value_loss.backward()
        self.value_network_optimizer.step()

        # Actor
        policy = self.actor_network(observations)             # [B, A]
        dist = torch.distributions.Categorical(policy)
        log_probs = dist.log_prob(actions)                    # [B]
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages.detach()).mean() - 0.001 * entropy

        self.actor_network_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_network_optimizer.step()

    def evaluate(self, render=False):
        env = self.monitor_env if render else self.env

        out = env.reset()
        observation = out[0] if isinstance(out, tuple) else out

        reward_episode = 0
        done = False

        while not done:
            observation = torch.tensor(observation, dtype=torch.float32)

            with torch.no_grad():
                policy = self.actor_network(observation)
                action = torch.argmax(policy).item()

            step_out = env.step(action)
            if len(step_out) == 5:
                observation, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                observation, reward, done, info = step_out

            reward_episode += reward

        return reward_episode
```

## Cell 18

```python
#@title Config A2C {display-mode: "form", run: "auto"}

env_id = 'CartPole-v1'  #@param ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
value_learning_rate = 0.001  #@param {type: "number"}
actor_learning_rate = 0.001  #@param {type: "number"}
gamma = 1  #@param {type: "number"}
entropy = 1  #@param {type: "number"}
seed = 1  #@param {type: "integer"}
#@markdown ---

config_a2c = {
    'env_id': env_id,
    'gamma': gamma,
    'seed': seed,
    'value_network': {'learning_rate': value_learning_rate},
    'actor_network': {'learning_rate': actor_learning_rate},
    'entropy': entropy
}

print("Current config_a2c is:")
pprint(config_a2c)
```

## Cell 19

```python
agent = A2CAgent(config_a2c)
rewards = agent.training_batch(1000, 256)
```

## Cell 20

```python
agent.evaluate(True)
```

