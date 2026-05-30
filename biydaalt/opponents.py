import numpy as np
from stable_baselines3 import PPO, A2C


def _constant_schedule(value):
    return lambda _progress_remaining: value


MODEL_COMPAT_OBJECTS = {
    "learning_rate": 3e-4,
    "lr_schedule": _constant_schedule(3e-4),
    "clip_range": _constant_schedule(0.2),
}


class RandomOpponent:
    def act(self, agent_name, obs, env):
        return env.action_space(agent_name).sample()


class PPOOpponent:
    def __init__(self, path):
        self.model = PPO.load(path, custom_objects=MODEL_COMPAT_OBJECTS)

    def act(self, agent_name, obs, env):
        action, _ = self.model.predict(
            np.expand_dims(obs, axis=0),
            deterministic=True
        )
        return int(action[0])


class A2COpponent:
    def __init__(self, path):
        self.model = A2C.load(path, custom_objects=MODEL_COMPAT_OBJECTS)

    def act(self, agent_name, obs, env):
        action, _ = self.model.predict(
            np.expand_dims(obs, axis=0),
            deterministic=True
        )
        return int(action[0])


class MixedOpponentPool:
    def __init__(self, opponents):
        self.opponents = opponents

    def sample(self):
        return np.random.choice(self.opponents)
