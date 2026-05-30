import os
import re
import argparse
from importlib.util import find_spec

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from battle_env import BattleV4CoalitionEnv, BattleV4StrategyEnv
from model import CoalitionSlotExtractor, StrategyMessageGRUExtractor
from opponents import RandomOpponent, PPOOpponent, A2COpponent, MODEL_COMPAT_OBJECTS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
TOTAL_TIMESTEPS = 2_000_000
TRAIN_CHUNK_TIMESTEPS = 500_000
CHECKPOINT_FREQ = 50_000
OPPONENT_CHECKPOINT_LIMIT = 4


def local_path(*parts):
    return os.path.join(BASE_DIR, *parts)


def get_tensorboard_log_dir():
    if find_spec("tensorboard") is None:
        print("TensorBoard is not installed; training will continue without TensorBoard logs.")
        return None

    return LOG_DIR


def run_name(strategy, env_type):
    if env_type == "single":
        return strategy
    if env_type == "coalition":
        return f"coalition_slot_{strategy}"

    return f"{env_type}_{strategy}"


def checkpoint_path(checkpoint_key, timesteps):
    return os.path.join(CHECKPOINT_DIR, f"ppo_{checkpoint_key}_{timesteps}_steps.zip")


def final_checkpoint_path(checkpoint_key):
    return os.path.join(CHECKPOINT_DIR, f"ppo_{checkpoint_key}_final.zip")


def extract_checkpoint_timesteps(path, checkpoint_key):
    filename = os.path.basename(path)

    match = re.fullmatch(rf"ppo_{re.escape(checkpoint_key)}_(\d+)_steps\.zip", filename)
    if match:
        return int(match.group(1))

    match = re.fullmatch(rf"ppo_{re.escape(checkpoint_key)}_stage_(\d+)\.zip", filename)
    if match:
        return int(match.group(1)) * TRAIN_CHUNK_TIMESTEPS

    return None


def find_latest_checkpoint(strategy, env_type="single"):
    checkpoints = get_strategy_checkpoints(run_name(strategy, env_type))
    if not checkpoints:
        return None

    return checkpoints[-1][1]


def get_strategy_checkpoints(checkpoint_key):
    checkpoints = []

    for filename in os.listdir(CHECKPOINT_DIR):
        if not filename.endswith(".zip"):
            continue

        path = os.path.join(CHECKPOINT_DIR, filename)
        timesteps = extract_checkpoint_timesteps(path, checkpoint_key)

        if timesteps is not None:
            checkpoints.append((timesteps, path))

    return sorted(checkpoints, key=lambda item: item[0])


def build_opponent_pool(strategy="swarming", opponent_mode="mixed", checkpoint_key=None):
    pool = []

    magent_path = local_path("magent2one4all.zip")

    if opponent_mode == "random":
        pool.append(RandomOpponent())
        print("Opponent pool size:", len(pool))
        return pool

    if opponent_mode in ("mixed", "selfplay"):
        pool.append(RandomOpponent())

    if opponent_mode in ("mixed", "magent2", "magent2-heavy"):
        if os.path.exists(magent_path):
            try:
                pool.append(PPOOpponent(magent_path))
                print("Loaded PPO opponent:", magent_path)
            except Exception as e:
                print("Failed loading opponent:", magent_path, e)

    if opponent_mode == "magent2-heavy":
        for _ in range(3):
            if os.path.exists(magent_path):
                try:
                    pool.append(PPOOpponent(magent_path))
                except Exception as e:
                    print("Failed loading extra magent2 opponent:", magent_path, e)

    candidates = [
        ("a2c", os.path.join(CHECKPOINT_DIR, "a2c_enemy.zip"))
    ]

    if checkpoint_key is None:
        checkpoint_key = strategy

    opponent_checkpoint_key = checkpoint_key
    if checkpoint_key.startswith("coalition_"):
        opponent_checkpoint_key = strategy

    if opponent_mode in ("mixed", "selfplay", "magent2-heavy"):
        for _timesteps, path in get_strategy_checkpoints(opponent_checkpoint_key)[-OPPONENT_CHECKPOINT_LIMIT:]:
            candidates.append(("ppo", path))

    for algo, path in candidates:
        if not os.path.exists(path):
            continue

        try:
            if algo == "ppo":
                pool.append(PPOOpponent(path))
                print("Loaded PPO opponent:", path)
            elif algo == "a2c":
                pool.append(A2COpponent(path))
                print("Loaded A2C opponent:", path)

        except Exception as e:
            print("Failed loading opponent:", path, e)

    print("Opponent pool size:", len(pool))
    return pool


def make_env(rank, strategy, opponent_mode, env_type, checkpoint_key):
    def _init():
        opponent_pool = build_opponent_pool(strategy, opponent_mode, checkpoint_key)

        if env_type == "coalition":
            env = BattleV4CoalitionEnv(
                map_size=45,
                max_cycles=500,
                render_mode=None,
                controlled_team="red",
                strategy=strategy,
                opponent_pool=opponent_pool,
                randomize_strategy=False,
            )
        else:
            env = BattleV4StrategyEnv(
                map_size=45,
                max_cycles=500,
                render_mode=None,
                controlled_team="red",
                strategy=strategy,
                opponent_pool=opponent_pool
            )

        env.reset(seed=rank)
        return Monitor(env)

    return _init


def make_training_env(strategy, opponent_mode, n_envs, env_type, checkpoint_key):
    if n_envs <= 1:
        return DummyVecEnv([make_env(0, strategy, opponent_mode, env_type, checkpoint_key)])

    return SubprocVecEnv(
        [
            make_env(rank, strategy, opponent_mode, env_type, checkpoint_key)
            for rank in range(n_envs)
        ],
        start_method="fork",
    )


def train_strategy(
    strategy="swarming",
    opponent_mode="mixed",
    n_envs=1,
    total_timesteps=TOTAL_TIMESTEPS,
    train_chunk_timesteps=TRAIN_CHUNK_TIMESTEPS,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    env_type="single",
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    checkpoint_key = run_name(strategy, env_type)
    env = make_training_env(strategy, opponent_mode, n_envs, env_type, checkpoint_key)

    if env_type == "coalition":
        policy_kwargs = dict(
            features_extractor_class=CoalitionSlotExtractor,
            features_extractor_kwargs=dict(
                features_dim=512,
                agent_hidden_dim=32,
                global_hidden_dim=128,
            ),
            net_arch=dict(
                pi=[512, 256],
                vf=[256, 128]
            )
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=StrategyMessageGRUExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
                hidden_dim=256,
                strategy_count=6,
                strategy_dim=32,
                message_dim=32
            ),
            net_arch=dict(
                pi=[256, 128],
                vf=[256, 128]
            )
        )

    latest_checkpoint = find_latest_checkpoint(strategy, env_type)

    if latest_checkpoint:
        print("Resuming from:", latest_checkpoint)
        model = PPO.load(
            latest_checkpoint,
            env=env,
            tensorboard_log=get_tensorboard_log_dir(),
            custom_objects=MODEL_COMPAT_OBJECTS,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=get_tensorboard_log_dir()
        )

    while model.num_timesteps < total_timesteps:
        current_timesteps = model.num_timesteps
        remaining_timesteps = total_timesteps - current_timesteps
        chunk_timesteps = min(train_chunk_timesteps, remaining_timesteps)
        next_target = current_timesteps + chunk_timesteps

        print("=" * 60)
        print(f"TRAINING STRATEGY: {strategy}")
        print(f"ENV TYPE: {env_type}")
        print(f"OPPONENT MODE: {opponent_mode}")
        print(f"PARALLEL ENVS: {n_envs}")
        print(f"N_STEPS: {n_steps}")
        print(f"BATCH SIZE: {batch_size}")
        print(f"N_EPOCHS: {n_epochs}")
        print(f"TIMESTEPS: {current_timesteps}/{total_timesteps}")
        print(f"NEXT TARGET: {next_target}/{total_timesteps}")
        print("=" * 60)

        save_freq = max(CHECKPOINT_FREQ // max(n_envs, 1), 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=CHECKPOINT_DIR,
            name_prefix=f"ppo_{checkpoint_key}",
        )

        model.learn(
            total_timesteps=chunk_timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"ppo_{checkpoint_key}",
            callback=checkpoint_callback
        )

        save_path = checkpoint_path(checkpoint_key, model.num_timesteps)
        model.save(save_path)
        print("Saved:", save_path)

    final_path = final_checkpoint_path(checkpoint_key)
    model.save(final_path)
    print("Final saved:", final_path)
    env.close()


def play(
    strategy="swarming",
    model_path=None,
    n_episodes=1,
    max_steps=500,
    render=False,
    deterministic=True,
):
    opponent_pool = build_opponent_pool(strategy, "mixed", run_name(strategy, "single"))

    env = BattleV4StrategyEnv(
        map_size=45,
        max_cycles=max_steps,
        render_mode="human" if render else None,
        controlled_team="red",
        strategy=strategy,
        opponent_pool=opponent_pool,
    )

    if model_path is None:
        model_path = find_latest_checkpoint(strategy)

    if model_path is None:
        raise FileNotFoundError(f"No checkpoint found for strategy: {strategy}")

    print("Playing with model:", model_path)
    model = PPO.load(
        model_path,
        env=env,
        custom_objects=MODEL_COMPAT_OBJECTS,
    )

    extractor = getattr(model.policy, "features_extractor", None)
    if hasattr(extractor, "set_strategy"):
        extractor.set_strategy(env.strategy_id)

    episode_results = []

    try:
        for episode in range(1, n_episodes + 1):
            obs, _info = env.reset()
            total_reward = 0.0
            last_info = {}

            for step in range(1, max_steps + 1):
                action, _state = model.predict(
                    obs,
                    deterministic=deterministic,
                )

                obs, reward, terminated, truncated, last_info = env.step(action)
                total_reward += reward

                if render:
                    env.render()

                if terminated or truncated:
                    break

            red_alive = last_info.get("red_alive", 0)
            blue_alive = last_info.get("blue_alive", 0)
            result = {
                "episode": episode,
                "steps": step,
                "reward": total_reward,
                "red_alive": red_alive,
                "blue_alive": blue_alive,
            }
            episode_results.append(result)

            print(
                "Episode {episode}: steps={steps}, reward={reward:.2f}, "
                "red_alive={red_alive}, blue_alive={blue_alive}".format(**result)
            )

    finally:
        env.close()

    return episode_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "play"], nargs="?", default="train")
    parser.add_argument("--strategy", default="swarming")
    parser.add_argument(
        "--opponent-mode",
        choices=["mixed", "magent2", "magent2-heavy", "selfplay", "random"],
        default="mixed",
    )
    parser.add_argument(
        "--env-type",
        choices=["single", "coalition"],
        default="single",
    )
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--chunk-timesteps", type=int, default=TRAIN_CHUNK_TIMESTEPS)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.mode == "play":
        play(
            strategy=args.strategy,
            model_path=args.model_path,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            render=args.render,
        )
    else:
        train_strategy(
            strategy=args.strategy,
            opponent_mode=args.opponent_mode,
            n_envs=args.n_envs,
            total_timesteps=args.total_timesteps,
            train_chunk_timesteps=args.chunk_timesteps,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            env_type=args.env_type,
        )
