"""
Message-Passing Multi-Agent A2C on MAgent2 Battle Environment
=============================================================
Each agent:
  1. Observes a local grid (obs)
  2. Receives aggregated messages from its k nearest neighbours
  3. Passes a message vector to its neighbours
  4. Actor-Critic network trained with shared weights (parameter sharing)

Rollout design
--------------
Agents die mid-episode, so the number alive per step is variable.
We store transitions as FLAT per-agent lists (one entry per agent per step)
rather than per-step arrays of shape (N_alive, ...).
GAE is computed per-agent trajectory using per-agent done flags.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.distributions import Categorical

from env_wrapper import MAgent2GymWrapper
from model import MessageA2CNetwork
from message_module import MessageAggregator
from utils import RunningMeanStd


# ─────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env",           type=str,   default="battle")
    p.add_argument("--team",          type=str,   default="red")
    p.add_argument("--n-episodes",    type=int,   default=2000)
    p.add_argument("--max-steps",     type=int,   default=500)
    p.add_argument("--n-steps",       type=int,   default=128,
                   help="Rollout transitions (summed across agents) before update")
    p.add_argument("--msg-dim",       type=int,   default=16)
    p.add_argument("--k-neighbors",   type=int,   default=4)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae-lambda",    type=float, default=0.95)
    p.add_argument("--value-coef",    type=float, default=0.5)
    p.add_argument("--entropy-coef",  type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden-dim",    type=int,   default=256)
    p.add_argument("--device",        type=str,   default="cpu")
    p.add_argument("--log-interval",  type=int,   default=20)
    p.add_argument("--save-dir",      type=str,   default="checkpoints")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────
# Rollout buffer — flat per-transition storage
# ─────────────────────────────────────────────
class RolloutBuffer:
    """
    Stores transitions as flat lists; each index = one (agent, step) pair.
    This handles variable numbers of alive agents per step naturally.
    """
    def __init__(self):
        self.obs      = []   # np.ndarray flat obs
        self.msgs     = []   # np.ndarray aggregated message
        self.actions  = []   # int
        self.rewards  = []   # float
        self.values   = []   # float  (from critic)
        self.log_probs= []   # float
        self.dones    = []   # bool

    def add(self, obs, msg, action, reward, value, log_prob, done):
        self.obs.append(obs)
        self.msgs.append(msg)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.obs)

    def clear(self):
        self.__init__()

    def compute_advantages_and_returns(self, last_value: float,
                                        gamma: float, lam: float):
        """
        Flat GAE over the stored transitions (treated as one long trajectory).
        `last_value` is the bootstrap value after the final transition.
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        # Append bootstrap value for the last step
        values_ext = self.values + [last_value]

        for t in reversed(range(T)):
            mask   = 1.0 - float(self.dones[t])
            delta  = self.rewards[t] + gamma * values_ext[t+1] * mask - values_ext[t]
            gae    = delta + gamma * lam * mask * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def to_tensors(self, device):
        obs_t    = torch.FloatTensor(np.stack(self.obs)).to(device)
        msgs_t   = torch.FloatTensor(np.stack(self.msgs)).to(device)
        acts_t   = torch.LongTensor(self.actions).to(device)
        logp_t   = torch.FloatTensor(self.log_probs).to(device)
        return obs_t, msgs_t, acts_t, logp_t


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    env = MAgent2GymWrapper(env_name=args.env, team=args.team,
                            max_steps=args.max_steps)
    obs_shape = env.obs_shape
    n_actions = env.n_actions
    obs_dim   = int(np.prod(obs_shape))
    print(f"[ENV]  obs_shape={obs_shape}  obs_dim={obs_dim}  n_actions={n_actions}")

    model = MessageA2CNetwork(
        obs_shape  = obs_shape,
        n_actions  = n_actions,
        msg_dim    = args.msg_dim,
        hidden_dim = args.hidden_dim,
    ).to(device)

    aggregator = MessageAggregator(msg_dim=args.msg_dim, k=args.k_neighbors)
    optimizer  = optim.Adam(model.parameters(), lr=args.lr)

    episode_rewards = []
    actor_losses    = []
    critic_losses   = []
    entropy_vals    = []
    win_rates       = []

    buf = RolloutBuffer()

    for episode in range(1, args.n_episodes + 1):
        obs_dict  = env.reset()
        positions = env.get_positions()
        msg_state = {aid: np.zeros(args.msg_dim, dtype=np.float32)
                     for aid in obs_dict}

        ep_reward = 0.0
        step      = 0
        done_all  = False

        while not done_all and step < args.max_steps:
            agent_ids = list(obs_dict.keys())
            if not agent_ids:
                break

            # ── Message aggregation ───────────────
            agg_msgs = aggregator.aggregate(agent_ids, positions, msg_state)

            obs_arr = np.stack([obs_dict[a] for a in agent_ids])   # (N, obs_dim)
            msg_arr = np.stack([agg_msgs[a] for a in agent_ids])   # (N, msg_dim)

            obs_t = torch.FloatTensor(obs_arr).to(device)
            msg_t = torch.FloatTensor(msg_arr).to(device)

            with torch.no_grad():
                logits, values, new_msgs = model(obs_t, msg_t)
                dist      = Categorical(logits=logits)
                actions   = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_np  = actions.cpu().numpy()
            values_np   = values.cpu().numpy().flatten()
            logp_np     = log_probs.cpu().numpy()
            new_msgs_np = new_msgs.cpu().numpy()

            # Update outgoing messages
            for i, aid in enumerate(agent_ids):
                msg_state[aid] = new_msgs_np[i]

            # Step environment
            action_dict = {aid: int(actions_np[i]) for i, aid in enumerate(agent_ids)}
            next_obs_dict, reward_dict, done_dict, _, positions = env.step(action_dict)

            step_reward = np.mean(list(reward_dict.values())) if reward_dict else 0.0
            ep_reward  += step_reward

            # ── Store one transition per agent ────
            for i, aid in enumerate(agent_ids):
                buf.add(
                    obs      = obs_arr[i],
                    msg      = msg_arr[i],
                    action   = int(actions_np[i]),
                    reward   = reward_dict.get(aid, 0.0),
                    value    = float(values_np[i]),
                    log_prob = float(logp_np[i]),
                    done     = done_dict.get(aid, False),
                )

            obs_dict = next_obs_dict
            done_all = (not obs_dict) or (bool(done_dict) and all(done_dict.values()))
            step    += 1

            # ── Update every n_steps transitions ─
            if len(buf) >= args.n_steps:
                last_val = _bootstrap_value(model, obs_dict, msg_state,
                                            args.msg_dim, device)
                a_loss, c_loss, ent = _update(model, optimizer, buf,
                                               last_val, args, device)
                actor_losses.append(a_loss)
                critic_losses.append(c_loss)
                entropy_vals.append(ent)
                buf.clear()

        # ── Final update ──────────────────────────
        if len(buf) > 0:
            last_val = _bootstrap_value(model, obs_dict, msg_state,
                                        args.msg_dim, device)
            a_loss, c_loss, ent = _update(model, optimizer, buf,
                                           last_val, args, device)
            actor_losses.append(a_loss)
            critic_losses.append(c_loss)
            entropy_vals.append(ent)
            buf.clear()

        episode_rewards.append(ep_reward)
        win_rates.append(env.get_win_rate())

        if episode % args.log_interval == 0:
            w  = args.log_interval
            mr  = np.mean(episode_rewards[-w:])
            mwr = np.mean(win_rates[-w:]) * 100
            mal = np.mean(actor_losses[-w:])  if actor_losses else 0.0
            mcl = np.mean(critic_losses[-w:]) if critic_losses else 0.0
            men = np.mean(entropy_vals[-w:])  if entropy_vals  else 0.0
            print(f"Ep {episode:5d}/{args.n_episodes} | "
                  f"Reward {mr:7.2f} | WinRate {mwr:5.1f}% | "
                  f"ActorL {mal:.4f} | CriticL {mcl:.4f} | Entropy {men:.3f}")

        if episode % 200 == 0:
            path = os.path.join(args.save_dir, f"model_ep{episode}.pt")
            torch.save({"episode": episode,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict()}, path)
            print(f"  ✓ Checkpoint → {path}")

    env.close()
    _plot_results(episode_rewards, win_rates, actor_losses, critic_losses, entropy_vals)
    print("\nTraining complete.")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _bootstrap_value(model, obs_dict, msg_state, msg_dim, device) -> float:
    """Estimate V(s_last) for GAE bootstrap. Returns 0 if no agents left."""
    if not obs_dict:
        return 0.0
    aid   = next(iter(obs_dict))
    obs_t = torch.FloatTensor(obs_dict[aid]).unsqueeze(0).to(device)
    msg_t = torch.FloatTensor(
        msg_state.get(aid, np.zeros(msg_dim, dtype=np.float32))
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        _, val, _ = model(obs_t, msg_t)
    return float(val.item())


def _update(model, optimizer, buf: RolloutBuffer,
            last_value: float, args, device):
    advantages, returns = buf.compute_advantages_and_returns(
        last_value, args.gamma, args.gae_lambda)

    obs_t, msgs_t, acts_t, _ = buf.to_tensors(device)
    adv_t = torch.FloatTensor(advantages).to(device)
    ret_t = torch.FloatTensor(returns).to(device)

    # Normalise advantages
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    logits, values, _ = model(obs_t, msgs_t)
    dist      = Categorical(logits=logits)
    log_probs = dist.log_prob(acts_t)
    entropy   = dist.entropy().mean()

    actor_loss  = -(log_probs * adv_t).mean()
    critic_loss = 0.5 * (ret_t - values.squeeze(-1)).pow(2).mean()
    total_loss  = (actor_loss
                   + args.value_coef  * critic_loss
                   - args.entropy_coef * entropy)

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    return actor_loss.item(), critic_loss.item(), entropy.item()


def _plot_results(rewards, win_rates, actor_losses, critic_losses, entropy):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable ({e}), skipping.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Message-Passing A2C on MAgent2")
    axes[0,0].plot(rewards);            axes[0,0].set_title("Episode Reward")
    axes[0,1].plot([w*100 for w in win_rates]); axes[0,1].set_title("Win Rate (%)")
    axes[1,0].plot(actor_losses,  label="Actor")
    axes[1,0].plot(critic_losses, label="Critic"); axes[1,0].legend()
    axes[1,0].set_title("Losses")
    axes[1,1].plot(entropy);            axes[1,1].set_title("Policy Entropy")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    print("Curves saved → training_curves.png")


if __name__ == "__main__":
    args = get_args()
    train(args)