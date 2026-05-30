import argparse
import os
import sys

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from mappo_like_env import BattleV4MAPPOParallelEnv
from opponents import MODEL_COMPAT_OBJECTS
from train import CHECKPOINT_DIR, LOG_DIR, build_opponent_pool, get_tensorboard_log_dir


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WARM_START_PATH = os.path.join(BASE_DIR, "magent2one4all.zip")


def checkpoint_key(strategy):
    return f"mappo_like_{strategy}"


def checkpoint_path(strategy, timesteps):
    return os.path.join(CHECKPOINT_DIR, f"ppo_{checkpoint_key(strategy)}_{timesteps}_steps.zip")


def final_checkpoint_path(strategy):
    return os.path.join(CHECKPOINT_DIR, f"ppo_{checkpoint_key(strategy)}_final.zip")


def make_parallel_env(strategy, opponent_mode, max_steps, render_mode=None):
    opponent_pool = build_opponent_pool(
        strategy=strategy,
        opponent_mode=opponent_mode,
        checkpoint_key=checkpoint_key(strategy),
    )
    env = BattleV4MAPPOParallelEnv(
        map_size=45,
        max_cycles=max_steps,
        render_mode=render_mode,
        controlled_team="red",
        opponent_pool=opponent_pool,
        reward_mode=strategy,
    )
    return ss.black_death_v3(env)


def make_vec_env(strategy, opponent_mode, max_steps, copies, num_cpus):
    env = make_parallel_env(strategy, opponent_mode, max_steps)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    return ss.concat_vec_envs_v1(
        vec_env,
        copies,
        num_cpus=num_cpus,
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
            obs, _infos = env.reset(seed=10_000 + episode)
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
                f"reward={episode_reward:.2f}"
            )
    finally:
        env.close()

    return {
        "episodes": episodes,
        "avg_reward": total_reward / max(episodes, 1),
        "survival_rate": total_survival_rate / max(episodes, 1),
        "avg_red_alive": total_red_alive / max(episodes, 1),
        "avg_blue_alive": total_blue_alive / max(episodes, 1),
        "timeouts": timeouts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="flee")
    parser.add_argument("--opponent-mode", default="mixed")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--chunk-timesteps", type=int, default=100_000)
    parser.add_argument("--n-env-copies", type=int, default=1)
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--warm-start", default=WARM_START_PATH)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if args.n_env_copies > 1:
        print(
            "WARNING: MAgent can segfault with Supersuit env copies in this setup. "
            "Use --n-env-copies 1 if python3.9 crashes."
        )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = make_vec_env(
        strategy=args.strategy,
        opponent_mode=args.opponent_mode,
        max_steps=args.max_steps,
        copies=args.n_env_copies,
        num_cpus=args.num_cpus,
    )

    load_path = args.resume or args.warm_start
    if load_path:
        print("Loading initial policy:", load_path)
        model = PPO.load(
            load_path,
            env=env,
            custom_objects=MODEL_COMPAT_OBJECTS,
            tensorboard_log=get_tensorboard_log_dir(),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device=args.device,
        )
        if args.resume is None:
            model.num_timesteps = 0
    else:
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
            device=args.device,
            tensorboard_log=get_tensorboard_log_dir(),
        )

    while model.num_timesteps < args.total_timesteps:
        remaining = args.total_timesteps - model.num_timesteps
        chunk = min(args.chunk_timesteps, remaining)
        print("=" * 60)
        print(f"MAPPO-LIKE TRAINING strategy={args.strategy}")
        print(f"opponent_mode={args.opponent_mode}")
        print(f"timesteps={model.num_timesteps}/{args.total_timesteps}")
        print(f"next_chunk={chunk}")
        print("=" * 60)

        callback = CheckpointCallback(
            save_freq=max(50_000 // max(args.n_env_copies, 1), 1),
            save_path=CHECKPOINT_DIR,
            name_prefix=f"ppo_{checkpoint_key(args.strategy)}",
        )
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            tb_log_name=f"ppo_{checkpoint_key(args.strategy)}",
            callback=callback,
        )

        save_path = checkpoint_path(args.strategy, model.num_timesteps)
        model.save(save_path)
        print("Saved:", save_path)

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
            f"avg_reward={metrics['avg_reward']:.2f}"
        )

    final_path = final_checkpoint_path(args.strategy)
    model.save(final_path)
    print("Final saved:", final_path)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
