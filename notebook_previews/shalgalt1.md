# shalgalt1.ipynb

Source notebook: [`shalgalt1.ipynb`](../shalgalt1.ipynb)

## Cell 1

```python
import random
import math
import numpy as np
from collections import deque, namedtuple

import gymnasium as gym
import ale_py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
```

## Cell 2

```python
try:
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
    USE_NEW_FRAMESTACK = True
except ImportError:
    from gymnasium.wrappers import AtariPreprocessing, FrameStack
    USE_NEW_FRAMESTACK = False


class FireResetEnv(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        try:
            meanings = self.env.unwrapped.get_action_meanings()
        except Exception:
            meanings = []

        if "FIRE" in meanings and self.env.action_space.n > 1:
            obs, _, terminated, truncated, info = self.env.step(1)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

            if self.env.action_space.n > 2:
                obs, _, terminated, truncated, info = self.env.step(2)
                if terminated or truncated:
                    obs, info = self.env.reset(**kwargs)

        return obs, info


def make_env(env_id="ALE/Pong-v5"):
    # frameskip=1 заавал хэрэгтэй.
    # AtariPreprocessing өөрөө frame_skip=4 хийдэг.
    env = gym.make(env_id, render_mode=None, frameskip=1)

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False
    )

    env = FireResetEnv(env)

    if USE_NEW_FRAMESTACK:
        env = FrameStackObservation(env, stack_size=4)
    else:
        env = FrameStack(env, num_stack=4)

    return env


env = make_env("ALE/Pong-v5")
n_actions = env.action_space.n

sample_obs, _ = env.reset()
sample_obs = np.array(sample_obs)

print("Observation shape:", sample_obs.shape)
print("Number of actions:", n_actions)
print("Action meanings:", env.unwrapped.get_action_meanings())

env.close()
```

## Cell 3

```python
def fix_obs_shape(obs):
    obs = np.array(obs)

    # Хэрвээ H x W x 4 хэлбэртэй байвал 4 x H x W болгоно
    if obs.ndim == 3 and obs.shape[-1] == 4:
        obs = np.transpose(obs, (2, 0, 1))

    # Хэрвээ ганц frame байвал 1 x H x W болгоно
    if obs.ndim == 2:
        obs = np.expand_dims(obs, axis=0)

    return obs


def obs_to_tensor(obs):
    obs = fix_obs_shape(obs)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    obs = obs / 255.0
    return obs.unsqueeze(0)


def get_last_frame(obs):
    obs = fix_obs_shape(obs)
    return obs[-1]


def longest_consecutive_segment(rows):
    if len(rows) == 0:
        return None

    best_start = rows[0]
    best_end = rows[0]
    cur_start = rows[0]
    cur_end = rows[0]

    for r in rows[1:]:
        if r == cur_end + 1:
            cur_end = r
        else:
            if cur_end - cur_start > best_end - best_start:
                best_start, best_end = cur_start, cur_end
            cur_start, cur_end = r, r

    if cur_end - cur_start > best_end - best_start:
        best_start, best_end = cur_start, cur_end

    return best_start, best_end


def paddle_y_from_obs(obs, side="left"):
    """
    Pong дээр opponent paddle-ийн y байрлалыг frame-ээс ойролцоогоор олно.
    side="left" гэдэг нь зүүн талын paddle-ийг өрсөлдөгч гэж үзэж байна.
    Хэрвээ буруу санагдвал side="right" болгож сольж болно.
    """
    frame = get_last_frame(obs)

    # score хэсгийг хасаж, зөвхөн тоглоомын талбайг харна
    y0, y1 = 18, 82

    if side == "left":
        x0, x1 = 5, 18
    else:
        x0, x1 = 66, 82

    crop = frame[y0:y1, x0:x1]

    # paddle нь цагаан/гэрэлтэй хэсэг байдаг
    mask = crop > 80

    row_counts = mask.sum(axis=1)
    rows = np.where(row_counts > 0)[0]

    segment = longest_consecutive_segment(rows)
    if segment is None:
        return None

    start, end = segment
    center_y = y0 + (start + end) / 2.0
    return center_y


def opponent_action_label(prev_obs, next_obs, side="left", deadzone=0.4):
    """
    Өрсөлдөгчийн paddle дээш/доош хөдөлсөн эсэхийг label болгоно.

    0 = stay
    1 = up
    2 = down
    """
    prev_y = paddle_y_from_obs(prev_obs, side=side)
    next_y = paddle_y_from_obs(next_obs, side=side)

    if prev_y is None or next_y is None:
        return 0

    dy = next_y - prev_y

    if dy < -deadzone:
        return 1   # up
    elif dy > deadzone:
        return 2   # down
    else:
        return 0   # stay
```

## Cell 4

```python
class AtariDQNWithOpponentPredictor(nn.Module):
    def __init__(self, input_shape, n_actions, n_opp_actions=3):
        super().__init__()

        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_size = self.conv(dummy).view(1, -1).size(1)

        self.feature = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )

        self.q_head = nn.Linear(512, n_actions)
        self.opp_head = nn.Linear(512, n_opp_actions)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.feature(x)

        q_values = self.q_head(x)
        opp_logits = self.opp_head(x)

        return q_values, opp_logits
```

## Cell 5

```python
Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done", "opp_label")
)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
```

## Cell 6

```python
env = make_env("ALE/Pong-v5")

sample_obs, _ = env.reset()
sample_obs = fix_obs_shape(sample_obs)

input_shape = sample_obs.shape
n_actions = env.action_space.n

print("Input shape:", input_shape)
print("Actions:", n_actions)

BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4

MEMORY_SIZE = 100_000
LEARNING_STARTS = 5_000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100_000

TARGET_UPDATE_EVERY = 10_000
TRAIN_EVERY = 4

NUM_EPISODES = 200
MAX_STEPS_PER_EPISODE = 10_000

OPPONENT_SIDE = "left"

# Зөв таамагласан үед нэмэх reward
OPP_PREDICT_BONUS = 0.05

# Opponent prediction loss-ийн жин
OPP_LOSS_COEF = 0.2

policy_net = AtariDQNWithOpponentPredictor(input_shape, n_actions).to(device)
target_net = AtariDQNWithOpponentPredictor(input_shape, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0

env_rewards = []
shaped_rewards = []
opponent_prediction_accs = []
losses = []
```

## Cell 7

```python
def select_action_and_predict_opponent(state):
    global steps_done

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )

    steps_done += 1

    with torch.no_grad():
        q_values, opp_logits = policy_net(state)
        predicted_opp_action = opp_logits.argmax(dim=1).item()

    if random.random() > eps_threshold:
        action = q_values.argmax(dim=1).item()
    else:
        action = env.action_space.sample()

    return action, predicted_opp_action


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_state_batch = torch.cat(batch.next_state)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device)
    opp_label_batch = torch.tensor(batch.opp_label, dtype=torch.long, device=device)

    q_values, opp_logits = policy_net(state_batch)
    current_q = q_values.gather(1, action_batch).squeeze(1)

    with torch.no_grad():
        next_q_values, _ = target_net(next_state_batch)
        next_q = next_q_values.max(dim=1)[0]
        expected_q = reward_batch + GAMMA * next_q * (1 - done_batch)

    dqn_loss = F.smooth_l1_loss(current_q, expected_q)
    opponent_prediction_loss = F.cross_entropy(opp_logits, opp_label_batch)

    loss = dqn_loss + OPP_LOSS_COEF * opponent_prediction_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()

    return loss.item()
```

## Cell 8

```python
for episode in range(1, NUM_EPISODES + 1):
    obs, info = env.reset()
    state = obs_to_tensor(obs)

    total_env_reward = 0
    total_shaped_reward = 0

    correct_pred = 0
    pred_count = 0

    for t in range(MAX_STEPS_PER_EPISODE):
        action, predicted_opp_action = select_action_and_predict_opponent(state)

        next_obs, env_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        true_opp_action = opponent_action_label(
            obs,
            next_obs,
            side=OPPONENT_SIDE
        )

        bonus = 0.0
        if predicted_opp_action == true_opp_action:
            bonus = OPP_PREDICT_BONUS
            correct_pred += 1

        pred_count += 1

        # Гол өөрчлөлт:
        # reward дээр өрсөлдөгчийн action зөв таамагласан bonus нэмнэ.
        shaped_reward = env_reward + bonus

        next_state = obs_to_tensor(next_obs)

        memory.push(
            state,
            action,
            shaped_reward,
            next_state,
            done,
            true_opp_action
        )

        state = next_state
        obs = next_obs

        total_env_reward += env_reward
        total_shaped_reward += shaped_reward

        if steps_done > LEARNING_STARTS and steps_done % TRAIN_EVERY == 0:
            loss = optimize_model()
            if loss is not None:
                losses.append(loss)

        if steps_done % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    acc = correct_pred / max(pred_count, 1)

    env_rewards.append(total_env_reward)
    shaped_rewards.append(total_shaped_reward)
    opponent_prediction_accs.append(acc)

    if episode % 10 == 0:
        print(
            f"Episode {episode:4d} | "
            f"Env reward: {total_env_reward:6.1f} | "
            f"Shaped reward: {total_shaped_reward:7.2f} | "
            f"Opp pred acc: {acc:.3f} | "
            f"Steps: {steps_done}"
        )

env.close()

torch.save(policy_net.state_dict(), "pong_dqn_opponent_prediction.pth")
print("Saved: pong_dqn_opponent_prediction.pth")
```

## Cell 9

```python
def moving_average(data, window=10):
    data = np.array(data, dtype=np.float32)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


plt.figure(figsize=(10, 5))
plt.plot(env_rewards, label="Original env reward")
plt.plot(shaped_rewards, label="Modified reward with opponent prediction bonus")

ma = moving_average(shaped_rewards, window=10)
plt.plot(
    range(len(ma)),
    ma,
    label="Moving average modified reward"
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN on Atari Pong with Opponent Action Prediction Bonus")
plt.legend()
plt.grid(True)
plt.show()
```

## Cell 10

```python
plt.figure(figsize=(10, 5))
plt.plot(opponent_prediction_accs, label="Opponent action prediction accuracy")

ma_acc = moving_average(opponent_prediction_accs, window=10)
plt.plot(
    range(len(ma_acc)),
    ma_acc,
    label="Moving average accuracy"
)

plt.xlabel("Episode")
plt.ylabel("Accuracy")
plt.title("Opponent Action Prediction Accuracy")
plt.legend()
plt.grid(True)
plt.show()
```

