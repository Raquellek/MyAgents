import argparse
import glob
import os
import sys


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from mappo_like_env import BattleV4MAPPOParallelEnv
from opponents import MODEL_COMPAT_OBJECTS
import train


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_WORKING = "/kaggle/working"
DEFAULT_OUTPUT_DIR = (
    os.path.join(KAGGLE_WORKING, "biydaalt_outputs")
    if os.path.exists(KAGGLE_WORKING)
    else os.path.join(BASE_DIR, "kaggle_outputs")
)


def find_warm_start():
    candidates = [
        os.path.join(BASE_DIR, "magent2one4all.zip"),
        os.path.join(os.getcwd(), "magent2one4all.zip"),
    ]
    candidates.extend(glob.glob("/kaggle/input/**/magent2one4all.zip", recursive=True))

    for path in candidates:
        if path and os.path.exists(path):
            return path

    return None


def configure_output_dirs(output_dir):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train.CHECKPOINT_DIR = checkpoint_dir
    train.LOG_DIR = log_dir
    return checkpoint_dir, log_dir


def checkpoint_key(strategy):
    return f"kaggle_mappo_like_{strategy}"


def make_parallel_env(strategy, opponent_mode, max_steps):
    opponent_pool = train.build_opponent_pool(
        strategy=strategy,
        opponent_mode=opponent_mode,
        checkpoint_key=checkpoint_key(strategy),
    )
    env = BattleV4MAPPOParallelEnv(
        map_size=45,
        max_cycles=max_steps,
        render_mode=None,
        controlled_team="red",
        opponent_pool=opponent_pool,
        reward_mode=strategy,
    )
    return ss.black_death_v3(env)


def make_vec_env(strategy, opponent_mode, max_steps):
    env = make_parallel_env(strategy, opponent_mode, max_steps)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    return ss.concat_vec_envs_v1(
        vec_env,
        1,
        num_cpus=0,
        base_class="stable_baselines3",
    )


def evaluate(model, strategy, opponent_mode, episodes, max_steps):
    env = make_parallel_env(strategy, opponent_mode, max_steps)

    total_reward = 0.0
    total_survival_rate = 0.0
    total_red_alive = 0
    total_blue_alive = 0
    timeouts = 0

    try:
        for episode in range(1, episodes + 1):
            obs, _infos = env.reset(seed=20_000 + episode)
            episode_reward = 0.0
            last_info = {}
            step = 0

            for step in range(1, max_steps + 1):
                actions = {}
                for agent, agent_obs in obs.items():
                    action, _state = model.predict(agent_obs, deterministic=True)
                    actions[agent] = int(action)

                obs, rewards, terminations, truncations, infos = env.step(actions)
                episode_reward += sum(rewards.values()) / max(len(rewards), 1)

                if infos:
                    last_info = next(iter(infos.values()))

                if not env.agents or all(terminations.values()) or all(truncations.values()):
                    break

            red_alive = last_info.get("red_alive", 0)
            blue_alive = last_info.get("blue_alive", 0)
            survival_rate = last_info.get("survival_rate", 0.0)

            total_reward += episode_reward
            total_survival_rate += survival_rate
            total_red_alive += red_alive
            total_blue_alive += blue_alive
            if step >= max_steps and red_alive > 0:
                timeouts += 1

            print(
                f"eval episode={episode} red={red_alive} blue={blue_alive} "
                f"survival_rate={survival_rate:.2f} steps={step} "
                f"reward={episode_reward:.2f}",
                flush=True,
            )
    finally:
        env.close()

    return {
        "episodes": episodes,
        "survival_rate": total_survival_rate / max(episodes, 1),
        "timeouts": timeouts,
        "avg_red_alive": total_red_alive / max(episodes, 1),
        "avg_blue_alive": total_blue_alive / max(episodes, 1),
        "avg_reward": total_reward / max(episodes, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="flee")
    parser.add_argument("--opponent-mode", default="random")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--chunk-timesteps", type=int, default=50_000)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warm-start", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    checkpoint_dir, log_dir = configure_output_dirs(args.output_dir)
    warm_start = args.warm_start if args.warm_start is not None else find_warm_start()

    print("Output dir:", args.output_dir, flush=True)
    print("Checkpoint dir:", checkpoint_dir, flush=True)
    print("Log dir:", log_dir, flush=True)
    print("Warm start:", warm_start, flush=True)
    print("Device:", args.device, flush=True)
    print("MAgent copies: 1 (Kaggle-safe)", flush=True)

    env = make_vec_env(
        strategy=args.strategy,
        opponent_mode=args.opponent_mode,
        max_steps=args.max_steps,
    )

    load_path = args.resume or warm_start
    if load_path:
        print("Loading initial policy:", load_path, flush=True)
        model = PPO.load(
            load_path,
            env=env,
            custom_objects=MODEL_COMPAT_OBJECTS,
            tensorboard_log=log_dir,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device=args.device,
        )
        if args.resume is None:
            model.num_timesteps = 0
    else:
        print("No warm-start found; training from scratch.", flush=True)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            device=args.device,
        )

    while model.num_timesteps < args.total_timesteps:
        remaining = args.total_timesteps - model.num_timesteps
        chunk = min(args.chunk_timesteps, remaining)

        print("=" * 60, flush=True)
        print(f"KAGGLE MAPPO-LIKE strategy={args.strategy}", flush=True)
        print(f"opponent_mode={args.opponent_mode}", flush=True)
        print(f"timesteps={model.num_timesteps}/{args.total_timesteps}", flush=True)
        print(f"next_chunk={chunk}", flush=True)
        print("=" * 60, flush=True)

        callback = CheckpointCallback(
            save_freq=50_000,
            save_path=checkpoint_dir,
            name_prefix=f"ppo_{checkpoint_key(args.strategy)}",
        )
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name=f"ppo_{checkpoint_key(args.strategy)}",
            callback=callback,
        )

        save_path = os.path.join(
            checkpoint_dir,
            f"ppo_{checkpoint_key(args.strategy)}_{model.num_timesteps}_steps.zip",
        )
        model.save(save_path)
        print("Saved:", save_path, flush=True)

        metrics = evaluate(
            model=model,
            strategy=args.strategy,
            opponent_mode=args.opponent_mode,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
        )
        print(
            "EVAL SUMMARY: "
            f"survival_rate={metrics['survival_rate']:.2f} "
            f"timeouts={metrics['timeouts']}/{metrics['episodes']} "
            f"avg_red_alive={metrics['avg_red_alive']:.1f} "
            f"avg_blue_alive={metrics['avg_blue_alive']:.1f} "
            f"avg_reward={metrics['avg_reward']:.2f}",
            flush=True,
        )

    final_path = os.path.join(
        checkpoint_dir,
        f"ppo_{checkpoint_key(args.strategy)}_final.zip",
    )
    model.save(final_path)
    print("Final saved:", final_path, flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
