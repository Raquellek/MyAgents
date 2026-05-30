import time
import os
import numpy as np
from stable_baselines3 import PPO, A2C
from magent2.environments import battle_v4


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "ppo_swarming_final.zip")
OPPONENT_PATH = os.path.join(BASE_DIR, "magent2one4all.zip")


def unpack_reset(reset_result):
    if isinstance(reset_result, tuple):
        obs, _infos = reset_result
        return obs

    return reset_result


def load_model(path):
    try:
        model = PPO.load(path)
        print("Loaded PPO:", path)
        return model
    except Exception as e:
        print("PPO load failed:", e)

    try:
        model = A2C.load(path)
        print("Loaded A2C:", path)
        return model
    except Exception as e:
        print("A2C load failed:", e)

    raise RuntimeError("Could not load model")


def main():
    env = battle_v4.parallel_env(
        render_mode="human",
        map_size=45,
        max_cycles=500
    )

    obs = unpack_reset(env.reset(seed=42))

    red_model = load_model(MODEL_PATH)
    blue_model = load_model(OPPONENT_PATH)

    episode = 0

    while True:
        actions = {}

        for agent in env.agents:
            ob = obs[agent]

            if agent.startswith("red"):
                action, _ = red_model.predict(
                    np.expand_dims(ob, axis=0),
                    deterministic=True
                )
                actions[agent] = int(action[0])

            else:
                action, _ = blue_model.predict(
                    np.expand_dims(ob, axis=0),
                    deterministic=True
                )
                actions[agent] = int(action[0])

        obs, rewards, terminations, truncations, infos = env.step(actions)

        env.render()

        red_alive = len([a for a in env.agents if a.startswith("red")])
        blue_alive = len([a for a in env.agents if a.startswith("blue")])

        print(
            f"\rEpisode {episode} | Red: {red_alive} | Blue: {blue_alive}",
            end=""
        )

        done = red_alive == 0 or blue_alive == 0 or len(env.agents) == 0

        if done:
            episode += 1
            print("\nEpisode finished")
            print("Red alive:", red_alive)
            print("Blue alive:", blue_alive)
            obs = unpack_reset(env.reset())

        time.sleep(0.01)


if __name__ == "__main__":
    main()
