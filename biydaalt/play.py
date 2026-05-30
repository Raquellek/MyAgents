import argparse
import os
import time

import numpy as np
from magent2.environments import battle_v4
from stable_baselines3 import A2C, PPO

from battle_env import BattleV4CoalitionEnv
from battle_env import STRATEGIES
from opponents import MODEL_COMPAT_OBJECTS
from opponents import PPOOpponent
from train import find_latest_checkpoint


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


AGENT1 = {
    "name": "agent1",
    "team": "red",
    "algo": "ppo",
    "path": "/home/margd/2026_HICHEEL/Agent/biydaalt/checkpoints/ppo_coalition_swarming_2230272_steps.zip",
    "architecture": "auto-detected",
    "env_type": "coalition",
}

AGENT2 = {
    "name": "agent2",
    "team": "blue",
    "algo": "ppo",
    "path": os.path.join(BASE_DIR, "magent2one4all.zip"),
    "architecture": "MlpPolicy",
    "env_type": "single",
}


def unpack_reset(reset_result):
    if isinstance(reset_result, tuple):
        obs, _infos = reset_result
        return obs

    return reset_result


def resolve_agent_path(agent, strategy):
    if agent["path"] is not None:
        return agent["path"]

    latest_checkpoint = find_latest_checkpoint(strategy, agent["env_type"])
    if latest_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found for strategy: {strategy}")

    return latest_checkpoint


def load_agent(agent, strategy):
    path = resolve_agent_path(agent, strategy)

    if not os.path.exists(path):
        raise FileNotFoundError(f"{agent['name']} model not found: {path}")

    algo = agent["algo"].lower()

    if algo == "ppo":
        model = PPO.load(path, custom_objects=MODEL_COMPAT_OBJECTS)
    elif algo == "a2c":
        model = A2C.load(path, custom_objects=MODEL_COMPAT_OBJECTS)
    else:
        raise ValueError(f"Unsupported algo for {agent['name']}: {agent['algo']}")

    extractor = getattr(model.policy, "features_extractor", None)
    if hasattr(extractor, "set_strategy"):
        extractor.set_strategy(STRATEGIES.get(strategy, STRATEGIES["swarming"]))

    print("=" * 60)
    print(f"{agent['name']} loaded")
    print(f"team: {agent['team']}")
    print(f"algo: {agent['algo'].upper()}")
    print(f"path: {path}")
    print(f"policy: {model.policy.__class__.__name__}")
    print(f"features_extractor: {extractor.__class__.__name__}")

    return model


def predict_action(model, obs, deterministic=True):
    action, _state = model.predict(
        np.expand_dims(obs, axis=0),
        deterministic=deterministic,
    )
    return int(action[0])


def play_single(
    strategy="swarming",
    episodes=1,
    max_steps=500,
    render=True,
    sleep=0.01,
    deterministic=True,
):
    agent1_model = load_agent(AGENT1, strategy)
    agent2_model = load_agent(AGENT2, strategy)

    models_by_team = {
        AGENT1["team"]: agent1_model,
        AGENT2["team"]: agent2_model,
    }

    env = battle_v4.parallel_env(
        render_mode="human" if render else None,
        map_size=45,
        max_cycles=max_steps,
    )

    try:
        for episode in range(1, episodes + 1):
            obs = unpack_reset(env.reset(seed=episode))
            total_rewards = {
                AGENT1["name"]: 0.0,
                AGENT2["name"]: 0.0,
            }

            for step in range(1, max_steps + 1):
                actions = {}

                for env_agent in env.agents:
                    team = env_agent.split("_", 1)[0]
                    model = models_by_team[team]
                    actions[env_agent] = predict_action(
                        model,
                        obs[env_agent],
                        deterministic=deterministic,
                    )

                obs, rewards, terminations, truncations, _infos = env.step(actions)

                for env_agent, reward in rewards.items():
                    team = env_agent.split("_", 1)[0]
                    if team == AGENT1["team"]:
                        total_rewards[AGENT1["name"]] += reward
                    elif team == AGENT2["team"]:
                        total_rewards[AGENT2["name"]] += reward

                if render:
                    env.render()
                    time.sleep(sleep)

                red_alive = len([a for a in env.agents if a.startswith("red")])
                blue_alive = len([a for a in env.agents if a.startswith("blue")])

                print(
                    f"\rEpisode {episode} | step={step} | "
                    f"red={red_alive} | blue={blue_alive}",
                    end="",
                )

                done = (
                    red_alive == 0
                    or blue_alive == 0
                    or len(env.agents) == 0
                    or all(terminations.values())
                    or all(truncations.values())
                )

                if done:
                    break

            print()
            print(
                f"Episode {episode} finished: "
                f"agent1_reward={total_rewards['agent1']:.2f}, "
                f"agent2_reward={total_rewards['agent2']:.2f}, "
                f"red_alive={red_alive}, blue_alive={blue_alive}"
            )

    finally:
        env.close()


def play_coalition(
    strategy="swarming",
    episodes=1,
    max_steps=500,
    render=True,
    sleep=0.01,
    deterministic=True,
):
    agent1_model = load_agent(AGENT1, strategy)

    if AGENT2["env_type"] == "coalition":
        raise ValueError(
            "BattleV4CoalitionEnv expects the opponent to act on one local "
            "agent observation at a time. Use a single-agent model for AGENT2."
        )

    agent2_path = resolve_agent_path(AGENT2, strategy)
    opponent = PPOOpponent(agent2_path)

    env = BattleV4CoalitionEnv(
        map_size=45,
        max_cycles=max_steps,
        render_mode="human" if render else None,
        controlled_team="red",
        strategy=strategy,
        opponent_pool=[opponent],
        randomize_strategy=False,
    )

    try:
        for episode in range(1, episodes + 1):
            obs, _info = env.reset(seed=episode)
            total_reward = 0.0
            last_info = {}

            for step in range(1, max_steps + 1):
                action, _state = agent1_model.predict(
                    obs,
                    deterministic=deterministic,
                )

                obs, reward, terminated, truncated, last_info = env.step(action)
                total_reward += reward

                if render:
                    env.render()
                    time.sleep(sleep)

                red_alive = last_info.get("red_alive", 0)
                blue_alive = last_info.get("blue_alive", 0)

                print(
                    f"\rEpisode {episode} | step={step} | "
                    f"red={red_alive} | blue={blue_alive}",
                    end="",
                )

                if terminated or truncated:
                    break

            print()
            print(
                f"Episode {episode} finished: "
                f"agent1_reward={total_reward:.2f}, "
                f"red_alive={last_info.get('red_alive', 0)}, "
                f"blue_alive={last_info.get('blue_alive', 0)}"
            )

    finally:
        env.close()


def play(
    strategy="swarming",
    episodes=1,
    max_steps=500,
    render=True,
    sleep=0.01,
    deterministic=True,
):
    if AGENT1["env_type"] == "coalition":
        play_coalition(
            strategy=strategy,
            episodes=episodes,
            max_steps=max_steps,
            render=render,
            sleep=sleep,
            deterministic=deterministic,
        )
        return

    play_single(
        strategy=strategy,
        episodes=episodes,
        max_steps=max_steps,
        render=render,
        sleep=sleep,
        deterministic=deterministic,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="swarming")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.01)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--agent1-env-type",
        choices=["single", "coalition"],
        default=AGENT1["env_type"],
    )
    parser.add_argument("--agent1-path", default=None)
    parser.add_argument("--agent2-path", default=None)
    args = parser.parse_args()

    if args.agent1_path is not None:
        AGENT1["path"] = args.agent1_path

    AGENT1["env_type"] = args.agent1_env_type
    if args.agent1_env_type == "coalition":
        AGENT1["architecture"] = "auto-detected"
    else:
        AGENT1["architecture"] = "auto-detected"

    if args.agent2_path is not None:
        AGENT2["path"] = args.agent2_path

    play(
        strategy=args.strategy,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=not args.no_render,
        sleep=args.sleep,
    )
