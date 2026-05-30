"""
evaluate.py
===========
Load a checkpoint and run evaluation episodes.
Optionally save the battle as:
  --save-tiff   →  multi-page TIFF  (one page per frame, lossless)
  --save-frames →  folder of PNG frames  (render/ep1/frame_0000.png ...)
  --save-video  →  MP4 video  (requires imageio[ffmpeg])

Usage examples:
    python evaluate.py --checkpoint checkpoints/model_ep200.pt
    python evaluate.py --checkpoint checkpoints/model_ep200.pt --save-tiff
    python evaluate.py --checkpoint checkpoints/model_ep200.pt --save-video
    python evaluate.py --checkpoint checkpoints/model_ep200.pt --save-frames
    python evaluate.py --checkpoint checkpoints/model_ep2000.pt \
        --save-tiff --save-video --n-episodes 3 --render-episode 1
"""

import argparse
import os
import numpy as np
import torch
from torch.distributions import Categorical

from env_wrapper import MAgent2GymWrapper
from model import MessageA2CNetwork
from message_module import MessageAggregator


# ─────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--env",             type=str, default="battle")
    p.add_argument("--team",            type=str, default="red")
    p.add_argument("--n-episodes",      type=int, default=10)
    p.add_argument("--max-steps",       type=int, default=500)
    p.add_argument("--msg-dim",         type=int, default=16)
    p.add_argument("--k-neighbors",     type=int, default=4)
    p.add_argument("--hidden-dim",      type=int, default=256)
    p.add_argument("--device",          type=str, default="cpu")
    # ── Rendering options ─────────────────────
    p.add_argument("--render-episode",  type=int, default=1,
                   help="Which episode number to render/save (default: 1)")
    p.add_argument("--save-tiff",       action="store_true",
                   help="Save rendered episode as a multi-page TIFF")
    p.add_argument("--save-frames",     action="store_true",
                   help="Save each frame as a PNG in render/epN/")
    p.add_argument("--save-video",      action="store_true",
                   help="Save rendered episode as MP4 (needs imageio[ffmpeg])")
    p.add_argument("--fps",             type=int, default=15,
                   help="Frames per second for video output")
    p.add_argument("--output-dir",      type=str, default="render",
                   help="Root directory for all render outputs")
    return p.parse_args()




# ─────────────────────────────────────────────
def save_tiff(frames, path):
    """Save list of (H,W,3) uint8 arrays as a multi-page TIFF."""
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        path,
        save_all    = True,
        append_images = imgs[1:],
        compression = "tiff_lzw",   # lossless LZW compression
    )
    print(f"  TIFF saved  → {path}  ({len(frames)} frames)")


def save_png_frames(frames, folder):
    """Save each frame as frame_NNNN.png inside folder."""
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(folder, f"frame_{i:04d}.png"))
    print(f"  PNG frames  → {folder}/  ({len(frames)} files)")


def save_video(frames, path, fps):
    """Save frames as MP4 using imageio (needs imageio[ffmpeg])."""
    try:
        import imageio
    except ImportError:
        print("  [video] imageio not found. Install with:  pip install imageio[ffmpeg]")
        return
    try:
        writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                    pixelformat="yuv420p", quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"  MP4 saved   → {path}  ({len(frames)} frames @ {fps} fps)")
    except Exception as e:
        # fallback to GIF if ffmpeg not available
        gif_path = path.replace(".mp4", ".gif")
        print(f"  [video] ffmpeg error ({e}), falling back to GIF → {gif_path}")
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"  GIF saved   → {gif_path}")


# ─────────────────────────────────────────────
def evaluate(args):
    device = torch.device(args.device)

    # ── Build env with render mode when we need frames ────────────────
    need_render = args.save_tiff or args.save_frames or args.save_video

    env = MAgent2GymWrapper(
        env_name     = args.env,
        team         = args.team,
        max_steps    = args.max_steps,
        render_mode  = "rgb_array" if need_render else None,
    )
    obs_shape = env.obs_shape
    n_actions = env.n_actions

    # ── Load model ────────────────────────────────────────────────────
    model = MessageA2CNetwork(
        obs_shape  = obs_shape,
        n_actions  = n_actions,
        msg_dim    = args.msg_dim,
        hidden_dim = args.hidden_dim,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ep_num = ckpt.get("episode", "?")
    print(f"Loaded checkpoint: {args.checkpoint}  (trained episode {ep_num})")

    aggregator  = MessageAggregator(msg_dim=args.msg_dim, k=args.k_neighbors)
    total_rewards = []
    win_rates     = []

    for ep in range(1, args.n_episodes + 1):
        capture_this = need_render and (ep == args.render_episode)

        obs_dict  = env.reset()
        positions = env.get_positions()
        msg_state = {aid: np.zeros(args.msg_dim, dtype=np.float32)
                     for aid in obs_dict}
        ep_reward = 0.0
        frames    = []

        # Capture first frame
        if capture_this:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        for step in range(args.max_steps):
            agent_ids = list(obs_dict.keys())
            if not agent_ids:
                break

            agg_msgs = aggregator.aggregate(agent_ids, positions, msg_state)
            obs_arr  = np.stack([obs_dict[a] for a in agent_ids])
            msg_arr  = np.stack([agg_msgs[a] for a in agent_ids])

            obs_t = torch.FloatTensor(obs_arr).to(device)
            msg_t = torch.FloatTensor(msg_arr).to(device)

            with torch.no_grad():
                logits, _, new_msgs = model(obs_t, msg_t)
                actions     = logits.argmax(dim=-1)   # greedy
                new_msgs_np = new_msgs.cpu().numpy()

            for i, aid in enumerate(agent_ids):
                msg_state[aid] = new_msgs_np[i]

            action_dict = {aid: int(actions[i].item())
                           for i, aid in enumerate(agent_ids)}
            obs_dict, reward_dict, done_dict, _, positions = env.step(action_dict)

            ep_reward += np.mean(list(reward_dict.values())) if reward_dict else 0.0

            if capture_this:
                frame = env.render()
            if frame is not None:
                frames.append(frame)

            if (not obs_dict) or (bool(done_dict) and all(done_dict.values())):
                break

        total_rewards.append(ep_reward)
        win_rates.append(env.get_win_rate())
        print(f"  Episode {ep:3d} | Reward {ep_reward:7.2f} | "
              f"WinRate {env.get_win_rate()*100:.1f}% | "
              f"Frames {len(frames) if capture_this else '-':>4}")

        # ── Save outputs for this episode ─────────────────────────────
        if capture_this and frames:
            os.makedirs(args.output_dir, exist_ok=True)
            ep_tag = f"ep{ep:03d}_trained{ep_num}"

            if args.save_tiff:
                tiff_path = os.path.join(args.output_dir, f"{ep_tag}.tiff")
                save_tiff(frames, tiff_path)

            if args.save_frames:
                png_dir = os.path.join(args.output_dir, ep_tag)
                save_png_frames(frames, png_dir)

            if args.save_video:
                mp4_path = os.path.join(args.output_dir, f"{ep_tag}.mp4")
                save_video(frames, mp4_path, args.fps)

    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  Mean Reward  : {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Mean WinRate : {np.mean(win_rates)*100:.1f}%")
    if need_render:
        print(f"  Render output→ {os.path.abspath(args.output_dir)}/")
    env.close()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    args = get_args()
    evaluate(args)