"""
Inference script for GRU-DQN battle agents.

Usage:
    python3 infer.py --red PATH_TO_RED.pt --blue PATH_TO_BLUE.pt [options]

Examples:
    # Run 3 episodes, save frames as video
    python3 infer.py --red dqn_gru_red_offensive_XXX_ep70.pt \
                     --blue dqn_gru_blue_defensive_XXX_ep70.pt \
                     --episodes 3 --video battle.mp4

    # Run 1 episode, print per-episode stats only
    python3 infer.py --red dqn_gru_red_offensive_XXX_ep70.pt \
                     --blue dqn_gru_blue_defensive_XXX_ep70.pt \
                     --episodes 1
"""

import argparse
import os
import random
from collections import deque

import numpy as np
import torch
from magent2.environments import battle_v4

# ── Import agent class from training file ─────────────────────────────────────
from strategy_dqn_gru import StrategicDQNAgent, _make_team_agents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_agent_from_checkpoint(ckpt_path, strategy_name, env, team_prefix,
                                sequence_length=4, hidden_dim=256):
    """Reconstruct a StrategicDQNAgent and load weights from a checkpoint."""
    first = next(a for a in env.possible_agents if a.startswith(team_prefix))
    agent = StrategicDQNAgent(
        input_shape=env.observation_spaces[first].shape,
        action_dim=env.action_spaces[first].n,
        strategy_name=strategy_name,
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        # Remaining hyperparams don't affect inference
        memory_size=1000,
        batch_size=128,
        epsilon_start=0.0,   # fully greedy
        epsilon_end=0.0,
    )
    ok = agent.load(ckpt_path)
    if not ok:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    agent.epsilon = 0.0      # force greedy regardless of saved epsilon
    agent.policy_net.eval()
    print(f"  Loaded {team_prefix} agent from {ckpt_path}  "
          f"(strategy={strategy_name}, saved ε={agent.epsilon:.4f} → forced 0)")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Run one episode
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(red_agent, blue_agent, sequence_length, render, frame_list):
    """
    Run one episode. Returns (red_total_reward, blue_total_reward, steps).
    If render=True, appends RGB frames to frame_list.
    """
    render_mode = "rgb_array" if render else None
    env = battle_v4.parallel_env(render_mode=render_mode)
    observations, _ = env.reset()

    state_buffers = {a: deque(maxlen=sequence_length) for a in env.possible_agents}
    red_r = blue_r = 0.0
    steps = 0

    while observations:
        if render:
            frame = env.render()
            if frame is not None:
                frame_list.append(frame)

        actions = {}
        for agent_name, obs in observations.items():
            team = red_agent if agent_name.startswith("red") else blue_agent
            actions[agent_name] = team.select_action_for_agent(
                obs, state_buffers[agent_name], training=False
            )

        next_obs, rewards, terminations, truncations, _ = env.step(actions)

        for agent_name, r in rewards.items():
            if agent_name.startswith("red"):
                red_r += r
            else:
                blue_r += r

        # Drop agents that are done
        observations = {
            a: o for a, o in next_obs.items()
            if not (terminations.get(a, False) or truncations.get(a, False))
        }
        steps += 1

    env.close()
    return red_r, blue_r, steps


# ─────────────────────────────────────────────────────────────────────────────
# Video saving (optional — requires imageio)
# ─────────────────────────────────────────────────────────────────────────────

def save_video(frames, path, fps=15):
    try:
        import imageio
    except ImportError:
        print("imageio not installed — skipping video save. Run: pip install imageio[ffmpeg]")
        return
    imageio.mimsave(path, frames, fps=fps)
    print(f"Video saved to {path}  ({len(frames)} frames @ {fps} fps)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inference for GRU-DQN battle agents")
    parser.add_argument("--red",  required=True, help="Path to red team checkpoint (.pt)")
    parser.add_argument("--blue", required=True, help="Path to blue team checkpoint (.pt)")
    parser.add_argument("--red-strategy",  default="offensive",
                        choices=["offensive", "defensive", "flanking", "swarming", "balanced"])
    parser.add_argument("--blue-strategy", default="defensive",
                        choices=["offensive", "defensive", "flanking", "swarming", "balanced"])
    parser.add_argument("--episodes",        type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--hidden-dim",      type=int, default=256)
    parser.add_argument("--video", default=None,
                        help="Save rendered frames to this video file (e.g. battle.mp4)")
    args = parser.parse_args()

    # Build a temporary env to get observation/action spaces
    _env = battle_v4.parallel_env(render_mode=None)
    _env.reset()

    print("\nLoading checkpoints...")
    red_agent  = load_agent_from_checkpoint(
        args.red,  args.red_strategy,  _env, "red_",
        args.sequence_length, args.hidden_dim,
    )
    blue_agent = load_agent_from_checkpoint(
        args.blue, args.blue_strategy, _env, "blue_",
        args.sequence_length, args.hidden_dim,
    )
    _env.close()

    render = args.video is not None
    frames = []
    all_red, all_blue = [], []

    print(f"\nRunning {args.episodes} episode(s)...")
    for ep in range(1, args.episodes + 1):
        red_r, blue_r, steps = run_episode(
            red_agent, blue_agent, args.sequence_length, render, frames
        )
        all_red.append(red_r)
        all_blue.append(blue_r)
        winner = "RED" if red_r > blue_r else ("BLUE" if blue_r > red_r else "DRAW")
        print(f"  Ep {ep:3d} | RED={red_r:8.1f}  BLUE={blue_r:8.1f}  steps={steps:4d}  → {winner}")

    print(f"\nSummary over {args.episodes} episodes:")
    print(f"  RED  avg={np.mean(all_red):8.2f}  min={np.min(all_red):8.2f}  max={np.max(all_red):8.2f}")
    print(f"  BLUE avg={np.mean(all_blue):8.2f}  min={np.min(all_blue):8.2f}  max={np.max(all_blue):8.2f}")
    red_wins  = sum(r > b for r, b in zip(all_red, all_blue))
    blue_wins = sum(b > r for r, b in zip(all_red, all_blue))
    print(f"  Wins: RED={red_wins}  BLUE={blue_wins}  DRAW={args.episodes - red_wins - blue_wins}")

    if render and frames:
        save_video(frames, args.video)


if __name__ == "__main__":
    main()
