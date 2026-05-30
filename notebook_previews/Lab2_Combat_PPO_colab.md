# Lab2_Combat_PPO_colab.ipynb

Source notebook: [`Lab2_Combat_PPO_colab.ipynb`](../Lab2_Combat_PPO_colab.ipynb)

# Lab 2 (Colab notebook ажиллана)
ma-gym-д PPO сургалт хийн олон агент тоглуулах.

Stable_baselines3-ий PPO-тай ижил боловч өөрчлөлт оруулах боломж бүхий сан PPO-PyTorch суулгах.

ma-gym Combat environment action_space болон observation_space нь агентын тоон хэмжээ бүхий class гэдгийг анхаараарай.

Даалгавар 1

нэг агент сурган түүнийг олон хувилан тоглуулах

Даалгавар 2

Олон агент зэрэг сурган тоглуулах

Хугацаа 4 долоо хоног

################################################################################
> # **Introduction**
> The notebook is divided into 5 major parts : 

*   **Part I** : define actor-critic network and PPO algorithm
*   **Part II** : train PPO algorithm and save network weights and log files
*   **Part III** : load (preTrained) network weights and test PPO algorithm
*   **Part IV** : load log files and plot graphs
*   **Part V** : install xvbf, load (preTrained) network weights and save images for gif and then generate gif

################################################################################

################################################################################
> # **Part - I**

*   define actor critic networks
*   define PPO algorithm

################################################################################

## Cell 4

```python
import gym
import sys
import pybullet as p

print("gym:", gym.__version__)
print("pybullet build:", p.getAPIVersion())   # эсвэл build time-ийг нь харна

try:
    import pybullet_envs
    print("pybullet_envs: OK")
except Exception as e:
    print("pybullet_envs: FAIL ->", type(e).__name__, e)
```

## Cell 5

```python


import os
import glob
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import gym
# import roboschool
import pybullet_envs

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
print("=" * 92)
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to:", torch.cuda.get_device_name(device))
else:
    print("Device set to: cpu")
print("=" * 92)


# ---------------------------------------------------------------------------
# RolloutBuffer  (Doc 1-ийн .clear() Pythonic хувилбар)
# ---------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.actions      = []
        self.states       = []
        self.logprobs     = []
        self.rewards      = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


# ---------------------------------------------------------------------------
# merge_buffers  (Doc 1-ээс авсан — multi-agent-д зайлшгүй)
# ---------------------------------------------------------------------------
def merge_buffers(buffers):
    """
    Agent бүрийн тусдаа RolloutBuffer-уудыг нэг buffer болгон нэгтгэнэ.
    Shared policy update-д ашиглана.
    """
    merged = RolloutBuffer()
    for buf in buffers:
        merged.states.extend(buf.states)
        merged.actions.extend(buf.actions)
        merged.logprobs.extend(buf.logprobs)
        merged.state_values.extend(buf.state_values)
        merged.rewards.extend(buf.rewards)
        merged.is_terminals.extend(buf.is_terminals)
    return merged


# ---------------------------------------------------------------------------
# ActorCritic  (Doc 2-ын continuous/discrete дэмжлэг + squeeze)
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init=0.6):
        super().__init__()
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full(
                (action_dim,), action_std_init ** 2
            ).to(device)

        # Actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64), nn.Tanh(),
                nn.Linear(64, 64),        nn.Tanh(),
                nn.Linear(64, action_dim),nn.Tanh(),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64), nn.Tanh(),
                nn.Linear(64, 64),        nn.Tanh(),
                nn.Linear(64, action_dim),nn.Softmax(dim=-1),
            )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),        nn.Tanh(),
            nn.Linear(64, 1),
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (self.action_dim,), new_action_std ** 2
            ).to(device)
        else:
            print("[WARNING] set_action_std() called on discrete policy — ignored.")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action          = dist.sample()
        action_logprob  = dist.log_prob(action)
        state_val       = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, states, actions):
        if self.has_continuous_action_space:
            action_mean = self.actor(states)
            action_var  = self.action_var.expand_as(action_mean)
            cov_mat     = torch.diag_embed(action_var).to(device)
            dist        = MultivariateNormal(action_mean, cov_mat)
            if self.action_dim == 1:
                actions = actions.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(states)
            dist         = Categorical(action_probs)

        action_logprobs = dist.log_prob(actions)
        dist_entropy    = dist.entropy()
        state_values    = torch.squeeze(self.critic(states))   # Doc 2-ын squeeze

        return action_logprobs, state_values, dist_entropy


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------
class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space=False,
        action_std_init=0.6,
        ent_coef=0.01,          # ← шинэ: entropy coefficient параметр болгосон
        max_grad_norm=0.5,      # ← шинэ: gradient clipping
    ):
        self.gamma                      = gamma
        self.eps_clip                   = eps_clip
        self.K_epochs                   = K_epochs
        self.has_continuous_action_space = has_continuous_action_space
        self.ent_coef                   = ent_coef
        self.max_grad_norm              = max_grad_norm

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.buffer     = RolloutBuffer()
        self.policy     = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
            {"params": self.policy.actor.parameters(),  "lr": lr_actor},
            {"params": self.policy.critic.parameters(), "lr": lr_critic},
        ])
        self.MseLoss = nn.MSELoss()

    # ------------------------------------------------------------------
    # Action std helpers  (Doc 2-оос)
    # ------------------------------------------------------------------
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("[WARNING] set_action_std() called on discrete policy — ignored.")

    def decay_action_std(self, decay_rate, min_std):
        if self.has_continuous_action_space:
            self.action_std = max(round(self.action_std - decay_rate, 4), min_std)
            self.set_action_std(self.action_std)
            print(f"action_std decayed to: {self.action_std}")
        else:
            print("[WARNING] decay_action_std() called on discrete policy — ignored.")

    # ------------------------------------------------------------------
    # select_action  (Doc 1-ийн buffer=None дэмжлэг + Doc 2-ын flatten)
    # ------------------------------------------------------------------
    def select_action(self, state, buffer=None):
        if buffer is None:
            buffer = self.buffer

        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state_t)

        buffer.states.append(state_t)
        buffer.actions.append(action)
        buffer.logprobs.append(action_logprob)
        buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()   # Doc 2
        else:
            return action.item()

    # ------------------------------------------------------------------
    # update  (advantage norm + gradient clipping нэмсэн)
    # ------------------------------------------------------------------
    def update(self):
        # Monte-Carlo returns
        rewards, discounted = [], 0.0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards),
            reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted = 0.0
            discounted = reward + self.gamma * discounted
            rewards.insert(0, discounted)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states       = torch.squeeze(torch.stack(self.buffer.states,       dim=0)).detach().to(device)
        old_actions      = torch.squeeze(torch.stack(self.buffer.actions,      dim=0)).detach().to(device)
        old_logprobs     = torch.squeeze(torch.stack(self.buffer.logprobs,     dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Advantage  ← шинэ: нормалчилсан
        advantages = rewards - old_state_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - self.ent_coef * dist_entropy      # параметр болгосон
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            # ← шинэ: gradient clipping
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path):
        torch.save(self.policy_old.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=device)
        self.policy_old.load_state_dict(state_dict)
        self.policy.load_state_dict(state_dict)
```

## Cell 6

```python



################################# End of Part I ################################


```

################################################################################
> # **Part - II**

*   train PPO algorithm on environments
*   save preTrained networks weights and log files

################################################################################

# PPO Сургалт хийх

## Cell 9

```python


# ============================================================
# Environment тохиргоо
# ============================================================
env_name                 = "Combat"
has_continuous_action_space = False          # Combat → discrete

max_ep_len               = 400              # нэг episode-ийн max timestep
max_training_timesteps   = int(1e5)         # нийт training timestep

print_freq               = max_ep_len * 4   # print interval (timestep)
log_freq                 = max_ep_len * 2   # log interval (timestep)
save_model_freq          = int(2e4)         # model хадгалах interval

action_std               = None             # discrete-д ашиглахгүй

# ============================================================
# PPO hyperparameters
# ============================================================
update_timestep = max_ep_len * 4   # policy update interval
K_epochs        = 10               # update бүрт хэдэн epoch (2e5-д 40 их)
eps_clip        = 0.2              # PPO clip parameter
gamma           = 0.99             # discount factor

lr_actor        = 0.0003           # actor learning rate
lr_critic       = 0.001            # critic learning rate

random_seed     = 0                # 0 = seed тогтоохгүй

# ============================================================
# Environment үүсгэх
# ============================================================
env = gym.make(
    'ma_gym:Combat-v0',
    grid_shape=(20, 20),
    n_agents=5,
    n_opponents=5,
    init_health=10,
    full_observable=False,
    step_cost=0,
    max_steps=100,
    step_cool=1,
)

state_dim  = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n if not has_continuous_action_space else env.action_space.shape[0]

print("=" * 92)
print("training environment name :", env_name)
print("state space dimension     :", state_dim)
print("action space dimension    :", action_dim)
print("n_agents                  :", env.n_agents)
print("has_continuous_action_space:", has_continuous_action_space)
print("-" * 92)
print("max training timesteps    :", max_training_timesteps)
print("max timesteps per episode :", max_ep_len)
print("PPO update frequency      :", update_timestep, "timesteps")
print("PPO K epochs              :", K_epochs)
print("PPO epsilon clip          :", eps_clip)
print("discount factor (gamma)   :", gamma)
print("lr_actor                  :", lr_actor)
print("lr_critic                 :", lr_critic)
print("=" * 92)

# ============================================================
# Logging тохиргоо  (Doc 4-ын exist_ok=True)
# ============================================================
log_dir = os.path.join("PPO_logs", env_name)
os.makedirs(log_dir, exist_ok=True)

run_num      = len(next(os.walk(log_dir))[2])
log_f_name   = os.path.join(log_dir, f"PPO_{env_name}_log_{run_num}.csv")
print("logging at:", log_f_name)

# ============================================================
# Checkpoint тохиргоо
# ============================================================
ckpt_dir        = os.path.join("PPO_preTrained", env_name)
os.makedirs(ckpt_dir, exist_ok=True)
checkpoint_path = os.path.join(ckpt_dir, f"PPO_{env_name}_{random_seed}_{run_num}.pth")
print("checkpoint path:", checkpoint_path)

# ============================================================
# Random seed  (Doc 3-оос)
# ============================================================
if random_seed:
    import torch
    print("setting random seed to", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

# ============================================================
# Shared PPO agent үүсгэх
# has_continuous_action_space зөв дамжуулсан  ← Doc 4-ын алдагдсан параметр
# ============================================================
shared_policy = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    lr_actor=lr_actor,
    lr_critic=lr_critic,
    gamma=gamma,
    K_epochs=K_epochs,
    eps_clip=eps_clip,
    has_continuous_action_space=has_continuous_action_space,  # ← зөв дамжуулсан
    action_std_init=0.6 if has_continuous_action_space else 0.0,
)

# ============================================================
# Training loop
# ============================================================
start_time = datetime.now().replace(microsecond=0)
print("Started training at:", start_time)
print("=" * 92)

log_f = open(log_f_name, "w+")
log_f.write("episode,timestep,reward\n")

print_running_reward   = 0.0
print_running_episodes = 0
log_running_reward     = 0.0
log_running_episodes   = 0

time_step = 0
i_episode = 0

while time_step <= max_training_timesteps:

    state = env.reset()
    current_ep_reward = 0.0

    # Agent бүрийн тусдаа rollout buffer  (Doc 4-ын зөв хэрэгжүүлэлт)
    agent_buffers = [RolloutBuffer() for _ in range(env.n_agents)]

    for t in range(1, max_ep_len + 1):

        # Нэг policy, тусдаа buffer → shared policy
        actions = [
            shared_policy.select_action(state[i], agent_buffers[i])
            for i in range(env.n_agents)
        ]

        next_state, reward, done_n, _ = env.step(actions)

        for i in range(env.n_agents):
            agent_buffers[i].rewards.append(reward[i])
            agent_buffers[i].is_terminals.append(done_n[i])

        state      = next_state
        time_step += 1
        current_ep_reward += np.sum(reward)

        # Policy update
        if time_step % update_timestep == 0:
            shared_policy.buffer = merge_buffers(agent_buffers)
            shared_policy.update()
            agent_buffers = [RolloutBuffer() for _ in range(env.n_agents)]

        # Continuous-д action std decay  (Doc 3-оос)
        if has_continuous_action_space and time_step % update_timestep == 0:
            pass  # action_std_decay_freq тохируулбал энд нэмнэ

        # Logging
        if time_step % log_freq == 0 and log_running_episodes > 0:
            log_avg_reward = round(log_running_reward / log_running_episodes, 4)
            log_f.write(f"{i_episode},{time_step},{log_avg_reward}\n")
            log_f.flush()
            log_running_reward   = 0.0
            log_running_episodes = 0

        # Print
        if time_step % print_freq == 0 and print_running_episodes > 0:
            print_avg_reward = round(print_running_reward / print_running_episodes, 4)
            print(f"Episode: {i_episode:5d} | Timestep: {time_step:7d} | Avg Reward: {print_avg_reward:.4f}")
            print_running_reward   = 0.0
            print_running_episodes = 0

        # Save
        if time_step % save_model_freq == 0:
            print(f"  → saving model at: {checkpoint_path}")
            shared_policy.save(checkpoint_path)
            print("     Elapsed:", datetime.now().replace(microsecond=0) - start_time)

        if all(done_n):
            break

    # Episode дуусахад үлдэгдэл rollout байвал update  (Doc 4-оос)
    leftover = sum(len(buf.rewards) for buf in agent_buffers)
    if leftover > 0:
        shared_policy.buffer = merge_buffers(agent_buffers)
        shared_policy.update()

    print_running_reward   += current_ep_reward
    print_running_episodes += 1
    log_running_reward     += current_ep_reward
    log_running_episodes   += 1
    i_episode              += 1

log_f.close()
shared_policy.save(checkpoint_path)
env.close()

print("=" * 92)
end_time = datetime.now().replace(microsecond=0)
print("Started training at :", start_time)
print("Finished training at:", end_time)
print("Total training time :", end_time - start_time)
print("=" * 92)
```

## Cell 10

```python



################################ End of Part II ################################


```

################################################################################
> # **Part - III**

*   load and test preTrained networks on environments

################################################################################

## Cell 12

```python
import gym

print("============================================================================================")

env_name = "Combat"
max_ep_len = 400
total_test_episodes = 10

K_epochs = 40
eps_clip = 0.2
gamma = 0.99
lr_actor = 0.0003
lr_critic = 0.001

random_seed = 0
run_num_pretrained = 0   # хэрэгтэй бол өөрчил

env = gym.make(
    'ma_gym:Combat-v0',
    grid_shape=(20,20),
    n_agents=5,
    n_opponents=5,
    init_health=10,
    full_observable=False,
    step_cost=0,
    max_steps=100,
    step_cool=1
)

state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n

shared_policy = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    lr_actor=lr_actor,
    lr_critic=lr_critic,
    gamma=gamma,
    K_epochs=K_epochs,
    eps_clip=eps_clip
)

checkpoint_path = f"PPO_preTrained/{env_name}/PPO_{env_name}_{random_seed}_{run_num_pretrained}.pth"
print("loading network from:", checkpoint_path)
shared_policy.load(checkpoint_path)

print("--------------------------------------------------------------------------------------------")

test_running_reward = 0

for ep in range(1, total_test_episodes + 1):
    ep_reward = 0
    state = env.reset()

    for t in range(1, max_ep_len + 1):
        actions = [shared_policy.select_action(state[i]) for i in range(env.n_agents)]
        state, reward, done_n, _ = env.step(actions)

        ep_reward += float(np.mean(reward))

        if all(done_n):
            break

    shared_policy.buffer.clear()

    test_running_reward += ep_reward
    print(f"Episode: {ep} \t Reward: {round(ep_reward, 4)}")

env.close()

print("============================================================================================")
avg_test_reward = round(test_running_reward / total_test_episodes, 4)
print("average test reward:", avg_test_reward)
print("============================================================================================")
```

## Cell 13

```python



################################ End of Part III ###############################


```

################################################################################
> # **Part - IV**

*   load log files using pandas
*   plot graph using matplotlib

################################################################################

## Cell 15

```python
import os
import glob
import gym
import matplotlib.pyplot as plt
from PIL import Image
from IPython import display as ipythondisplay

try:
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(400, 300))
    display.start()
    use_virtual_display = True
except Exception:
    use_virtual_display = False

print("============================================================================================")

env_name = "Combat"
max_ep_len = 400
total_test_episodes = 1

K_epochs = 40
eps_clip = 0.2
gamma = 0.99
lr_actor = 0.0003
lr_critic = 0.001

random_seed = 0
run_num_pretrained = 0   # хэрэгтэй бол өөрчил
render_ipython = True

env = gym.make(
    'ma_gym:Combat-v0',
    grid_shape=(20,20),
    n_agents=5,
    n_opponents=5,
    init_health=10,
    full_observable=False,
    step_cost=0,
    max_steps=100,
    step_cool=1
)

state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n

gif_images_dir = f"PPO_gif_images/{env_name}/"
gif_dir = f"PPO_gifs/{env_name}/"
os.makedirs(gif_images_dir, exist_ok=True)
os.makedirs(gif_dir, exist_ok=True)

# өмнөх frame-үүдийг арилгана
for f in glob.glob(os.path.join(gif_images_dir, "*.jpg")):
    os.remove(f)

shared_policy = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    lr_actor=lr_actor,
    lr_critic=lr_critic,
    gamma=gamma,
    K_epochs=K_epochs,
    eps_clip=eps_clip
)

checkpoint_path = f"PPO_preTrained/{env_name}/PPO_{env_name}_{random_seed}_{run_num_pretrained}.pth"
print("loading network from:", checkpoint_path)
shared_policy.load(checkpoint_path)

print("--------------------------------------------------------------------------------------------")

test_running_reward = 0
saved_frames = 0

for ep in range(1, total_test_episodes + 1):
    ep_reward = 0
    state = env.reset()

    for t in range(1, max_ep_len + 1):
        actions = [shared_policy.select_action(state[i]) for i in range(env.n_agents)]
        state, reward, done_n, _ = env.step(actions)

        ep_reward += float(np.mean(reward))

        img = env.render(mode="rgb_array")

        if render_ipython:
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis("off")
            ipythondisplay.clear_output(wait=True)
            ipythondisplay.display(plt.gcf())
            plt.close()

        Image.fromarray(img).save(os.path.join(gif_images_dir, str(t).zfill(6) + ".jpg"))
        saved_frames += 1

        if all(done_n):
            break

    shared_policy.buffer.clear()

    test_running_reward += ep_reward
    print(f"Episode: {ep} \t Reward: {round(ep_reward, 4)}")

env.close()

if render_ipython:
    ipythondisplay.clear_output(wait=True)

if use_virtual_display:
    display.stop()

print("============================================================================================")
print("total number of frames saved:", saved_frames)
avg_test_reward = round(test_running_reward / total_test_episodes, 4)
print("average test reward:", avg_test_reward)
print("============================================================================================")
```

## Cell 16

```python



################################ End of Part IV ################################


```

################################################################################
> # **Part - V**

*   install virtual display libraries for rendering on colab / remote server ^
*   load preTrained networks and save images for gif
*   generate and save gif from previously saved images

*   ^ If running locally; do not install xvbf and pyvirtualdisplay. Just comment out the virtual display code and render it normally. 
*   ^ You will still require to use ipythondisplay, if you want to render it in the Jupyter Notebook.

################################################################################

## Cell 18

```python
import os
import glob
from PIL import Image

print("============================================================================================")

env_name = "Combat"
gif_num = 0

total_timesteps = 300
step = 5
frame_duration = 150

gif_images_pattern = f"PPO_gif_images/{env_name}/*.jpg"
gif_dir = f"PPO_gifs/{env_name}"
os.makedirs(gif_dir, exist_ok=True)

gif_path = os.path.join(gif_dir, f"PPO_{env_name}_gif_{gif_num}.gif")

img_paths = sorted(glob.glob(gif_images_pattern))
img_paths = img_paths[:total_timesteps]
img_paths = img_paths[::step]

if len(img_paths) == 0:
    print("No images found for GIF generation.")
else:
    print("total frames in gif:", len(img_paths))
    print("total duration of gif:", round(len(img_paths) * frame_duration / 1000, 2), "seconds")

    img, *imgs = [Image.open(f) for f in img_paths]
    img.save(
        fp=gif_path,
        format="GIF",
        append_images=imgs,
        save_all=True,
        optimize=True,
        duration=frame_duration,
        loop=0
    )

    print("saved gif at:", gif_path)

print("============================================================================================")
```

## Cell 19

```python



################################# End of Part V ################################


```

################################################################################

---------------------------------------------------------------------------- That's all folks ! ----------------------------------------------------------------------------


################################################################################

## Cell 21

```python
from queue import Queue
from typing import Any, Dict, List
import time
import threading

class Message:
    def __init__(self, sender: str, receiver: str, content: Any):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.timestamp = time.time()

    def info(self, own_loc:Any, opp_loc:Any):
        self.own_loc = own_loc
        self.opp_loc = opp_loc

    def reject(self, reason:str):
        self.content = f"Message rejected: {reason}"   
    
    def accept(self):
        self.content = "Message accept"

    def acknowledge(self):
        self.content = "message acknowledge"

    def error(self, error_msg:str):
        self.content = f"Message rejected: {error_msg}"   

    
    

class MessageBroker:
    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.lock = threading.Lock()
    
    def register_agent(self, agent_id: str):
        with self.lock:
            if agent_id not in self.queues:
                self.queues[agent_id] = Queue()
    
    def send_message(self, message: Message):
        with self.lock:
            if message.receiver in self.queues:
                self.queues[message.receiver].put(message)
            else:
                raise ValueError(f"Хүлээн авагч олдсонгүй: {message.receiver}")
    
    def get_message(self, agent_id: str) -> Message:
        if agent_id in self.queues:
            return self.queues[agent_id].get()
        raise ValueError(f"Агент олдсонгүй: {agent_id}")

class Agent:
    def __init__(self, agent_id: str, broker: MessageBroker):
        self.agent_id = agent_id
        self.broker = broker
        self.broker.register_agent(agent_id)
        self.received_messages: List[Message] = []
        self._running = False
        self._thread = None
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._process_messages)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
    
    def send_message(self, message, receiver: str, content: Any):
        self.message = Message(self.agent_id, receiver, content)
        self.broker.send_message(self,message)
    
    def _process_messages(self):
        while self._running:
            try:
                message = self.broker.get_message(self.agent_id)
                self.received_messages.append(message)
                self._handle_message(message)
            except Exception as e:
                print(f"Алдаа гарлаа {self.agent_id}: {str(e)}")
    
    def _handle_message(self, message: Message):
        print(f"Агент {self.agent_id} хүлээн авсан мессеж: {message.content} (илгээгч: {message.sender})")

# Жишээ ашиглалт
def main():
    # Message broker үүсгэх
    broker = MessageBroker()
    
    # Агентууд үүсгэх
    agent1 = Agent("agent1", broker)
    agent2 = Agent("agent2", broker)
    
    # Агентуудыг эхлүүлэх
    agent1.start()
    agent2.start()
    
    # Мессеж илгээх
    agent1.send_message("agent2", "Сайн байна уу!")
    time.sleep(1)  # Хүлээх
    agent2.send_message("agent1", "Сайн, сайн!")
    time.sleep(1)
    agent1.send_message.acknowledge()
    
    # Агентуудыг зогсоох
    agent1.stop()
    agent2.stop()

if __name__ == "__main__":
    main()
```

