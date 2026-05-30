"""
infer_message.py — Inference for message-passing GRU-DQN agents
================================================================
Loads checkpoints saved by message_strategy_dqn_gru.py and runs
greedy rollouts with full communication (k-NN message aggregation).

Usage
-----
    python3 infer_message.py \
        --red  msg_dqn_red_offensive_TIMESTAMP_ep500.pt \
        --blue msg_dqn_blue_defensive_TIMESTAMP_ep500.pt \
        [--episodes 5] [--video battle.mp4] \
        [--red-strategy offensive] [--blue-strategy defensive] \
        [--msg-dim 16] [--k-neighbors 4] \
        [--hidden-dim 256] [--sequence-length 4]
"""

import argparse
import os
from collections import deque

import numpy as np
import torch
from magent2.environments import battle_v4

from message_strategy_dqn_gru import (
    MessageStrategicDQNAgent,
    get_team_positions,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_agent(ckpt_path, strategy_name, env, team_prefix,
               msg_dim, k_neighbors, hidden_dim, sequence_length):
    first = next(a for a in env.possible_agents if a.startswith(team_prefix))
    agent = MessageStrategicDQNAgent(
        input_shape     = env.observation_spaces[first].shape,
        action_dim      = env.action_spaces[first].n,
        strategy_name   = strategy_name,
        msg_dim         = msg_dim,
        k_neighbors     = k_neighbors,
        hidden_dim      = hidden_dim,
        sequence_length = sequence_length,
        memory_size     = 1000,   # irrelevant for inference
        epsilon_start   = 0.0,
        epsilon_end     = 0.0,
    )
    if not agent.load(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    agent.epsilon = 0.0          # fully greedy
    agent.policy_net.eval()
    saved_eps = torch.load(ckpt_path, map_location=device).get('epsilon', 0.0)
    print(f"  [{team_prefix}] loaded {os.path.basename(ckpt_path)} "
          f"strategy={strategy_name}  saved_ε={saved_eps:.4f} → forced 0")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Single episode rollout
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(red_agent, blue_agent, sequence_length, msg_dim, render, frames):
    """
    Run one greedy episode with full message passing.
    Appends RGB frames to `frames` if render=True.
    Returns (red_total_reward, blue_total_reward, n_steps).
    """
    render_mode = "rgb_array" if render else None
    env = battle_v4.parallel_env(render_mode=render_mode)
    observations, _ = env.reset()

    # Per-agent GRU state buffers
    state_buffers = {a: deque(maxlen=sequence_length) for a in env.possible_agents}
    msg_buffers   = {a: deque(maxlen=sequence_length) for a in env.possible_agents}

    # Per-team outgoing message state
    red_msg_state  = {a: np.zeros(msg_dim, np.float32)
                      for a in env.possible_agents if a.startswith("red")}
    blue_msg_state = {a: np.zeros(msg_dim, np.float32)
                      for a in env.possible_agents if a.startswith("blue")}

    red_r = blue_r = 0.0
    steps = 0

    while observations:
        if render:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        active     = list(observations.keys())
        red_active = [a for a in active if a.startswith("red")]
        blue_active= [a for a in active if a.startswith("blue")]

        # Aggregate neighbour messages per team
        red_pos  = get_team_positions(env, "red",  active)
        blue_pos = get_team_positions(env, "blue", active)
        red_agg  = red_agent.aggregate_messages(red_active,  red_pos,  red_msg_state)
        blue_agg = blue_agent.aggregate_messages(blue_active, blue_pos, blue_msg_state)

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
        steps += 1

    env.close()
    return red_r, blue_r, steps


# ─────────────────────────────────────────────────────────────────────────────
# Message stats (optional diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def run_episode_with_msg_stats(red_agent, blue_agent, sequence_length, msg_dim):
    """
    Like run_episode but also collects message statistics for analysis.
    Returns (red_r, blue_r, steps, msg_stats_dict).
    """
    env = battle_v4.parallel_env(render_mode=None)
    observations, _ = env.reset()

    state_buffers  = {a: deque(maxlen=sequence_length) for a in env.possible_agents}
    msg_buffers    = {a: deque(maxlen=sequence_length) for a in env.possible_agents}
    red_msg_state  = {a: np.zeros(msg_dim, np.float32)
                      for a in env.possible_agents if a.startswith("red")}
    blue_msg_state = {a: np.zeros(msg_dim, np.float32)
                      for a in env.possible_agents if a.startswith("blue")}

    red_r = blue_r = 0.0
    steps = 0
    all_red_msgs  = []
    all_blue_msgs = []

    while observations:
        active     = list(observations.keys())
        red_active = [a for a in active if a.startswith("red")]
        blue_active= [a for a in active if a.startswith("blue")]

        red_pos  = get_team_positions(env, "red",  active)
        blue_pos = get_team_positions(env, "blue", active)
        red_agg  = red_agent.aggregate_messages(red_active,  red_pos,  red_msg_state)
        blue_agg = blue_agent.aggregate_messages(blue_active, blue_pos, blue_msg_state)

        actions = {}
        for agent_name, obs in observations.items():
            if agent_name.startswith("red"):
                agg_msg = red_agg.get(agent_name, np.zeros(msg_dim, np.float32))
                action, new_msg = red_agent.select_action_for_agent(
                    obs, agg_msg, state_buffers[agent_name],
                    msg_buffers[agent_name], training=False,
                )
                red_msg_state[agent_name] = new_msg
                all_red_msgs.append(new_msg)
            else:
                agg_msg = blue_agg.get(agent_name, np.zeros(msg_dim, np.float32))
                action, new_msg = blue_agent.select_action_for_agent(
                    obs, agg_msg, state_buffers[agent_name],
                    msg_buffers[agent_name], training=False,
                )
                blue_msg_state[agent_name] = new_msg
                all_blue_msgs.append(new_msg)
            actions[agent_name] = action

        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        for agent_name, r in rewards.items():
            if agent_name.startswith("red"): red_r += r
            else:                            blue_r += r

        observations = {
            a: o for a, o in next_obs.items()
            if not (terminations.get(a, False) or truncations.get(a, False))
        }
        steps += 1

    env.close()

    msg_stats = {}
    for team, msgs in [("red", all_red_msgs), ("blue", all_blue_msgs)]:
        if msgs:
            arr = np.stack(msgs)
            msg_stats[team] = {
                "mean":   arr.mean(axis=0),
                "std":    arr.std(axis=0),
                "abs_mean": np.abs(arr).mean(),
                "n_transmissions": len(msgs),
            }
    return red_r, blue_r, steps, msg_stats


# ─────────────────────────────────────────────────────────────────────────────
# Video
# ─────────────────────────────────────────────────────────────────────────────

def save_video(frames, path, fps=15):
    try:
        import imageio
    except ImportError:
        print("imageio not installed — skip. Run: pip install imageio[ffmpeg]")
        return
    imageio.mimsave(path, frames, fps=fps)
    print(f"Video saved → {path}  ({len(frames)} frames @ {fps} fps)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference for message-passing GRU-DQN battle agents"
    )
    parser.add_argument("--red",  required=True, help="Red team checkpoint (.pt)")
    parser.add_argument("--blue", required=True, help="Blue team checkpoint (.pt)")
    parser.add_argument("--red-strategy",     default="offensive",
                        choices=["offensive","defensive","flanking","swarming","balanced"])
    parser.add_argument("--blue-strategy",    default="defensive",
                        choices=["offensive","defensive","flanking","swarming","balanced"])
    parser.add_argument("--episodes",         type=int,   default=5)
    parser.add_argument("--msg-dim",          type=int,   default=16)
    parser.add_argument("--k-neighbors",      type=int,   default=4)
    parser.add_argument("--hidden-dim",       type=int,   default=256)
    parser.add_argument("--sequence-length",  type=int,   default=4)
    parser.add_argument("--video",            default=None,
                        help="Save rendered episode to this mp4 file")
    parser.add_argument("--msg-stats",        action="store_true",
                        help="Print message vector statistics for the first episode")
    args = parser.parse_args()

    # Temporary env to read spaces
    _env = battle_v4.parallel_env(render_mode=None)
    _env.reset()

    print("\nLoading checkpoints...")
    red_agent  = load_agent(args.red,  args.red_strategy,  _env, "red_",
                            args.msg_dim, args.k_neighbors,
                            args.hidden_dim, args.sequence_length)
    blue_agent = load_agent(args.blue, args.blue_strategy, _env, "blue_",
                            args.msg_dim, args.k_neighbors,
                            args.hidden_dim, args.sequence_length)
    _env.close()

    # Optional: message stats on episode 1
    if args.msg_stats:
        print("\nCollecting message statistics (episode 1)...")
        r_r, b_r, steps, stats = run_episode_with_msg_stats(
            red_agent, blue_agent, args.sequence_length, args.msg_dim
        )
        print(f"  Episode 1: RED={r_r:.1f}  BLUE={b_r:.1f}  steps={steps}")
        for team, s in stats.items():
            print(f"  {team.upper()} messages: "
                  f"transmissions={s['n_transmissions']}  "
                  f"|msg|_mean={s['abs_mean']:.4f}  "
                  f"per-dim std={s['std'].mean():.4f}")

    render = args.video is not None
    frames = []
    all_red, all_blue = [], []

    print(f"\nRunning {args.episodes} episode(s) "
          f"[msg_dim={args.msg_dim}  k={args.k_neighbors}]...")
    for ep in range(1, args.episodes + 1):
        red_r, blue_r, steps = run_episode(
            red_agent, blue_agent,
            args.sequence_length, args.msg_dim,
            render and ep == 1,  # render only first episode
            frames,
        )
        all_red.append(red_r)
        all_blue.append(blue_r)
        winner = ("RED" if red_r > blue_r else
                  "BLUE" if blue_r > red_r else "DRAW")
        print(f"  Ep {ep:3d} | RED={red_r:8.1f}  BLUE={blue_r:8.1f}  "
              f"steps={steps:4d}  → {winner}")

    print(f"\nSummary over {args.episodes} episode(s):")
    print(f"  RED  avg={np.mean(all_red):8.2f}  "
          f"min={np.min(all_red):8.2f}  max={np.max(all_red):8.2f}")
    print(f"  BLUE avg={np.mean(all_blue):8.2f}  "
          f"min={np.min(all_blue):8.2f}  max={np.max(all_blue):8.2f}")
    red_wins  = sum(r > b for r, b in zip(all_red, all_blue))
    blue_wins = sum(b > r for r, b in zip(all_red, all_blue))
    draws     = args.episodes - red_wins - blue_wins
    print(f"  Wins: RED={red_wins}  BLUE={blue_wins}  DRAW={draws}")

    if render and frames:
        save_video(frames, args.video)


if __name__ == "__main__":
    main()
