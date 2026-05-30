import argparse
import os
import sys

from stable_baselines3 import PPO

from battle_env import BattleV4CoalitionEnv
from opponents import MODEL_COMPAT_OBJECTS
from train import (
    CHECKPOINT_DIR,
    build_opponent_pool,
    find_latest_checkpoint,
    get_strategy_checkpoints,
    run_name,
    train_strategy,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def latest_timesteps(checkpoint_key):
    checkpoints = get_strategy_checkpoints(checkpoint_key)
    if not checkpoints:
        return 0

    return checkpoints[-1][0]


def evaluate_coalition(
    model_path,
    strategy="flee",
    opponent_mode="mixed",
    episodes=5,
    max_steps=500,
):
    model = PPO.load(model_path, custom_objects=MODEL_COMPAT_OBJECTS)
    opponent_pool = build_opponent_pool(
        strategy=strategy,
        opponent_mode=opponent_mode,
        checkpoint_key=run_name(strategy, "coalition"),
    )
    env = BattleV4CoalitionEnv(
        map_size=45,
        max_cycles=max_steps,
        render_mode=None,
        controlled_team="red",
        strategy=strategy,
        opponent_pool=opponent_pool,
        randomize_strategy=False,
    )

    wins = 0
    draws = 0
    losses = 0
    survived = 0
    timeouts = 0
    total_reward = 0.0
    total_survival_rate = 0.0
    total_red_alive = 0
    total_blue_alive = 0

    try:
        for episode in range(1, episodes + 1):
            obs, _info = env.reset(seed=10_000 + episode)
            last_info = {}
            episode_reward = 0.0

            for _step in range(1, max_steps + 1):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, last_info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            red_alive = last_info.get("red_alive", 0)
            blue_alive = last_info.get("blue_alive", 0)
            survival_rate = last_info.get(
                "survival_rate",
                red_alive / max(env.max_controlled_agents, 1),
            )

            total_survival_rate += survival_rate
            total_red_alive += red_alive
            total_blue_alive += blue_alive

            if survival_rate >= 0.8:
                survived += 1
            if _step >= max_steps and red_alive > 0:
                timeouts += 1

            if red_alive > blue_alive:
                wins += 1
            elif red_alive == blue_alive:
                draws += 1
            else:
                losses += 1

            total_reward += episode_reward
            print(
                f"eval episode={episode} red={red_alive} blue={blue_alive} "
                f"survival_rate={survival_rate:.2f} steps={_step} "
                f"reward={episode_reward:.2f}"
            )

    finally:
        env.close()

    win_rate = wins / max(episodes, 1)
    survival_rate = total_survival_rate / max(episodes, 1)
    avg_reward = total_reward / max(episodes, 1)
    avg_red_alive = total_red_alive / max(episodes, 1)
    avg_blue_alive = total_blue_alive / max(episodes, 1)

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "survived": survived,
        "timeouts": timeouts,
        "episodes": episodes,
        "win_rate": win_rate,
        "survival_rate": survival_rate,
        "avg_reward": avg_reward,
        "avg_red_alive": avg_red_alive,
        "avg_blue_alive": avg_blue_alive,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="flee")
    parser.add_argument("--opponent-mode", default="mixed")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--chunk-timesteps", type=int, default=250_000)
    parser.add_argument("--max-total-timesteps", type=int, default=3_000_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--target-win-rate", type=float, default=0.6)
    parser.add_argument("--target-survival-rate", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint_key = run_name(args.strategy, "coalition")

    while True:
        model_path = find_latest_checkpoint(args.strategy, "coalition")

        if model_path is not None:
            print("=" * 60)
            print("EVALUATING:", model_path)
            metrics = evaluate_coalition(
                model_path=model_path,
                strategy=args.strategy,
                opponent_mode=args.opponent_mode,
                episodes=args.eval_episodes,
                max_steps=args.max_steps,
            )
            print(
                "EVAL SUMMARY: "
                f"wins={metrics['wins']}/{metrics['episodes']} "
                f"draws={metrics['draws']} losses={metrics['losses']} "
                f"win_rate={metrics['win_rate']:.2f} "
                f"survived={metrics['survived']}/{metrics['episodes']} "
                f"timeouts={metrics['timeouts']} "
                f"survival_rate={metrics['survival_rate']:.2f} "
                f"avg_red_alive={metrics['avg_red_alive']:.1f} "
                f"avg_blue_alive={metrics['avg_blue_alive']:.1f} "
                f"avg_reward={metrics['avg_reward']:.2f}"
            )

            if args.strategy == "flee":
                target_reached = metrics["survival_rate"] >= args.target_survival_rate
            else:
                target_reached = metrics["win_rate"] >= args.target_win_rate

            if target_reached:
                print("Target reached.")
                return 0

        current_timesteps = latest_timesteps(checkpoint_key)
        if current_timesteps >= args.max_total_timesteps:
            print(
                "Stopped before target: "
                f"current_timesteps={current_timesteps}, "
                f"max_total_timesteps={args.max_total_timesteps}"
            )
            return 1

        next_total = min(
            current_timesteps + args.chunk_timesteps,
            args.max_total_timesteps,
        )

        print("=" * 60)
        print(
            f"TRAINING NEXT CHUNK: {current_timesteps} -> {next_total} "
            f"against {args.opponent_mode}"
        )
        print("=" * 60)

        train_strategy(
            strategy=args.strategy,
            opponent_mode=args.opponent_mode,
            n_envs=args.n_envs,
            total_timesteps=next_total,
            train_chunk_timesteps=args.chunk_timesteps,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            env_type="coalition",
        )


if __name__ == "__main__":
    sys.exit(main())
