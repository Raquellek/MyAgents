import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from magent2.environments import battle_v4

from opponents import RandomOpponent


STRATEGIES = {
    "offensive": 0,
    "defensive": 1,
    "swarming": 2,
    "flanking": 3,
    "balanced": 4,
    "flee": 5,
}


STRATEGY_NAMES = list(STRATEGIES.keys())


class BattleV4StrategyEnv(gym.Env):
    """
    Single-policy PPO wrapper.
    Red team = сурах model.
    Blue team = opponent policy.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        map_size=45,
        max_cycles=500,
        render_mode=None,
        controlled_team="red",
        strategy="balanced",
        opponent_pool=None
    ):
        super().__init__()

        self.env = battle_v4.parallel_env(
            map_size=map_size,
            max_cycles=max_cycles,
            render_mode=render_mode
        )

        self.map_size = map_size
        self.max_cycles = max_cycles
        self.render_mode = render_mode

        self.controlled_team = controlled_team
        self.enemy_team = "blue" if controlled_team == "red" else "red"

        self.strategy = strategy
        self.strategy_id = STRATEGIES.get(strategy, 4)

        self.opponent_pool = opponent_pool or [RandomOpponent()]
        self.current_opponent = random.choice(self.opponent_pool)

        self.obs = self._reset_parallel_env(seed=42)

        first_agent = self.env.agents[0]
        self.max_controlled_agents = len(self.controlled_agents())
        self.local_obs_shape = self.obs[first_agent].shape
        self.local_obs_size = int(np.prod(self.local_obs_shape))
        self.action_n = self.env.action_space(first_agent).n
        self.tactical_feature_size = 6
        self.agent_feature_size = self.local_obs_size + self.tactical_feature_size

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.max_controlled_agents, self.agent_feature_size),
            dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete(
            [self.action_n] * self.max_controlled_agents
        )

    def _reset_parallel_env(self, seed=None):
        reset_result = self.env.reset(seed=seed)

        if isinstance(reset_result, tuple):
            obs, _infos = reset_result
            return obs

        return reset_result

    def controlled_agents(self):
        return [
            a for a in self.env.agents
            if a.startswith(self.controlled_team)
        ]

    def enemy_agents(self):
        return [
            a for a in self.env.agents
            if a.startswith(self.enemy_team)
        ]

    def _agent_index(self, agent_name):
        try:
            return int(agent_name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    def _build_observation(self):
        state = np.zeros(self.observation_space.shape, dtype=np.float32)

        controlled = self.controlled_agents()
        red_alive = len([a for a in self.env.agents if a.startswith("red")])
        blue_alive = len([a for a in self.env.agents if a.startswith("blue")])
        denom = max(self.max_controlled_agents, 1)

        if self.controlled_team == "red":
            ally_alive = red_alive
            enemy_alive = blue_alive
        else:
            ally_alive = blue_alive
            enemy_alive = red_alive

        advantage = (ally_alive - enemy_alive) / denom

        for slot, agent in enumerate(controlled[:self.max_controlled_agents]):
            local_obs = self.obs.get(agent)
            if local_obs is None:
                continue

            tactical = np.array(
                [
                    1.0,
                    self._agent_index(agent) / denom,
                    slot / denom,
                    ally_alive / denom,
                    enemy_alive / denom,
                    advantage,
                ],
                dtype=np.float32,
            )
            state[slot] = np.concatenate(
                [local_obs.astype(np.float32).reshape(-1), tactical]
            )

        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.obs = self._reset_parallel_env(seed=seed)
        self.current_opponent = random.choice(self.opponent_pool)

        return self._build_observation(), {}

    def reward_shape(self, raw_reward, red_alive, blue_alive, done):
        reward = raw_reward

        if self.controlled_team == "red":
            advantage = red_alive - blue_alive
        else:
            advantage = blue_alive - red_alive

        if self.strategy == "offensive":
            reward = reward * 1.5 + 0.03 * advantage

        elif self.strategy == "defensive":
            reward = reward * 0.9
            if not done:
                reward += 0.04
            if advantage < 0:
                reward -= 0.03

        elif self.strategy == "swarming":
            reward = reward * 1.2 + 0.05 * advantage

        elif self.strategy == "flanking":
            reward = reward * 1.3 + 0.02 * advantage

        elif self.strategy == "balanced":
            reward = reward + 0.02 * advantage

        return float(reward)

    def step(self, action):
        actions = {}
        action = np.asarray(action).reshape(-1)

        controlled = self.controlled_agents()
        enemies = self.enemy_agents()

        for slot, agent in enumerate(controlled[:self.max_controlled_agents]):
            if agent in self.obs:
                actions[agent] = int(action[slot])

        for agent in enemies:
            if agent in self.obs:
                actions[agent] = self.current_opponent.act(
                    agent,
                    self.obs[agent],
                    self.env
                )

        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

        red_alive = len([a for a in self.env.agents if a.startswith("red")])
        blue_alive = len([a for a in self.env.agents if a.startswith("blue")])

        total_reward = 0.0

        for agent in controlled:
            total_reward += rewards.get(agent, 0.0)

        done = (
            red_alive == 0
            or blue_alive == 0
            or len(self.env.agents) == 0
        )

        shaped_reward = self.reward_shape(
            total_reward,
            red_alive,
            blue_alive,
            done
        )

        self.obs = next_obs

        info = {
            "red_alive": red_alive,
            "blue_alive": blue_alive,
            "strategy": self.strategy,
            "raw_reward": total_reward
        }

        terminated = done
        truncated = False

        return self._build_observation(), shaped_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class BattleV4CoalitionEnv(gym.Env):
    """
    Team-level PPO wrapper.
    One policy outputs a separate action for each controlled agent slot.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        map_size=45,
        max_cycles=500,
        render_mode=None,
        controlled_team="red",
        strategy="swarming",
        opponent_pool=None,
        squad_count=4,
        randomize_strategy=True,
    ):
        super().__init__()

        self.env = battle_v4.parallel_env(
            map_size=map_size,
            max_cycles=max_cycles,
            render_mode=render_mode
        )

        self.map_size = map_size
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.controlled_team = controlled_team
        self.enemy_team = "blue" if controlled_team == "red" else "red"
        self.strategy = strategy
        self.randomize_strategy = randomize_strategy
        self.squad_count = squad_count
        self.opponent_pool = opponent_pool or [RandomOpponent()]
        self.current_opponent = random.choice(self.opponent_pool)

        self.obs = self._reset_parallel_env(seed=42)
        self.max_controlled_agents = len(self.controlled_agents())

        first_agent = self.env.agents[0]
        self.local_obs_shape = self.obs[first_agent].shape
        self.local_obs_size = int(np.prod(self.local_obs_shape))
        self.action_n = self.env.action_space(first_agent).n
        self.tactical_feature_size = 13
        self.agent_feature_size = self.local_obs_size + self.tactical_feature_size

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.max_controlled_agents, self.agent_feature_size),
            dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [self.action_n] * self.max_controlled_agents
        )

        self.strategy_id = STRATEGIES.get(strategy, STRATEGIES["swarming"])

    def _reset_parallel_env(self, seed=None):
        reset_result = self.env.reset(seed=seed)

        if isinstance(reset_result, tuple):
            obs, _infos = reset_result
            return obs

        return reset_result

    def controlled_agents(self):
        return [
            a for a in self.env.agents
            if a.startswith(self.controlled_team)
        ]

    def enemy_agents(self):
        return [
            a for a in self.env.agents
            if a.startswith(self.enemy_team)
        ]

    def _choose_strategy(self):
        if self.randomize_strategy:
            self.strategy = random.choice(STRATEGY_NAMES)

        self.strategy_id = STRATEGIES.get(self.strategy, STRATEGIES["swarming"])

    def _agent_index(self, agent_name):
        try:
            return int(agent_name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    def _squad_id(self, slot_index):
        if self.max_controlled_agents <= 1:
            return 0

        squad_size = max(1, int(np.ceil(self.max_controlled_agents / self.squad_count)))
        return min(slot_index // squad_size, self.squad_count - 1)

    def _density_features(self, local_obs):
        channels = local_obs.reshape(-1, local_obs.shape[-1])
        channel_means = channels.mean(axis=0)

        padded = np.zeros(5, dtype=np.float32)
        padded[:min(5, len(channel_means))] = channel_means[:5]

        return padded

    def _build_observation(self):
        state = np.zeros(self.observation_space.shape, dtype=np.float32)

        controlled = self.controlled_agents()
        red_alive = len([a for a in self.env.agents if a.startswith("red")])
        blue_alive = len([a for a in self.env.agents if a.startswith("blue")])
        denom = max(self.max_controlled_agents, 1)

        if self.controlled_team == "red":
            ally_alive = red_alive
            enemy_alive = blue_alive
        else:
            ally_alive = blue_alive
            enemy_alive = red_alive

        advantage = (ally_alive - enemy_alive) / denom

        for slot, agent in enumerate(controlled[:self.max_controlled_agents]):
            local_obs = self.obs.get(agent)
            if local_obs is None:
                continue

            flat_obs = local_obs.astype(np.float32).reshape(-1)
            density = self._density_features(local_obs)
            squad_id = self._squad_id(slot)
            agent_index = self._agent_index(agent)

            tactical = np.array(
                [
                    1.0,
                    agent_index / denom,
                    slot / denom,
                    squad_id / max(self.squad_count - 1, 1),
                    self.strategy_id / max(len(STRATEGIES) - 1, 1),
                    ally_alive / denom,
                    enemy_alive / denom,
                    advantage,
                    *density,
                ],
                dtype=np.float32,
            )

            state[slot] = np.concatenate([flat_obs, tactical])

        return state

    def reward_shape(
        self,
        raw_reward,
        red_alive,
        blue_alive,
        done,
        pre_red_alive=None,
        pre_blue_alive=None,
    ):
        if self.controlled_team == "red":
            ally_alive = red_alive
            enemy_alive = blue_alive
            pre_ally_alive = pre_red_alive
            pre_enemy_alive = pre_blue_alive
        else:
            ally_alive = blue_alive
            enemy_alive = red_alive
            pre_ally_alive = pre_blue_alive
            pre_enemy_alive = pre_red_alive

        if pre_ally_alive is None:
            pre_ally_alive = ally_alive
        if pre_enemy_alive is None:
            pre_enemy_alive = enemy_alive

        advantage = ally_alive - enemy_alive

        if self.strategy == "flee":
            ally_lost = max(pre_ally_alive - ally_alive, 0)
            enemy_lost = max(pre_enemy_alive - enemy_alive, 0)
            survival_ratio = ally_alive / max(self.max_controlled_agents, 1)

            reward = 1.0
            reward += 0.25 * raw_reward
            reward += 2.0 * survival_ratio
            reward += 0.05 * ally_alive
            reward += 4.0 * enemy_lost
            reward -= 8.0 * ally_lost

            if done:
                if ally_alive == 0:
                    reward -= 50.0
                elif enemy_alive == 0 and ally_alive > 0:
                    reward += 25.0 * survival_ratio
                else:
                    reward += 35.0 * survival_ratio

            return float(reward)

        reward = raw_reward + 0.12 * advantage
        reward += 0.03 * ally_alive
        reward -= 0.04 * enemy_alive

        if self.strategy == "offensive":
            reward = reward * 1.35
        elif self.strategy == "defensive":
            reward = reward * 0.95 + 0.03 * ally_alive
        elif self.strategy == "swarming":
            reward = reward * 1.2 + 0.04 * advantage
        elif self.strategy == "flanking":
            reward = reward * 1.25 + 0.02 * enemy_alive

        if done:
            if enemy_alive == 0 and ally_alive > 0:
                reward += 30.0 + 0.5 * ally_alive
            elif ally_alive == 0 and enemy_alive > 0:
                reward -= 30.0 + 0.5 * enemy_alive
            elif ally_alive == 0 and enemy_alive == 0:
                reward -= 15.0
            elif ally_alive > enemy_alive:
                reward += 20.0
            elif ally_alive < enemy_alive:
                reward -= 20.0

        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.obs = self._reset_parallel_env(seed=seed)
        self.current_opponent = random.choice(self.opponent_pool)
        self._choose_strategy()

        return self._build_observation(), {}

    def step(self, action):
        actions = {}
        action = np.asarray(action).reshape(-1)

        controlled = self.controlled_agents()
        enemies = self.enemy_agents()
        pre_red_alive = len([a for a in self.env.agents if a.startswith("red")])
        pre_blue_alive = len([a for a in self.env.agents if a.startswith("blue")])

        for slot, agent in enumerate(controlled[:self.max_controlled_agents]):
            if agent in self.obs:
                actions[agent] = int(action[slot])

        for agent in enemies:
            if agent in self.obs:
                actions[agent] = self.current_opponent.act(
                    agent,
                    self.obs[agent],
                    self.env
                )

        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

        red_alive = len([a for a in self.env.agents if a.startswith("red")])
        blue_alive = len([a for a in self.env.agents if a.startswith("blue")])
        all_truncated = len(truncations) > 0 and all(truncations.values())

        if all_truncated and len(self.env.agents) == 0:
            red_alive = pre_red_alive
            blue_alive = pre_blue_alive

        total_reward = 0.0
        for agent in controlled:
            total_reward += rewards.get(agent, 0.0)

        done = (
            red_alive == 0
            or blue_alive == 0
            or len(self.env.agents) == 0
            or (len(terminations) > 0 and all(terminations.values()))
            or all_truncated
        )

        self.obs = next_obs
        shaped_reward = self.reward_shape(
            total_reward,
            red_alive,
            blue_alive,
            done,
            pre_red_alive=pre_red_alive,
            pre_blue_alive=pre_blue_alive,
        )

        info = {
            "red_alive": red_alive,
            "blue_alive": blue_alive,
            "strategy": self.strategy,
            "raw_reward": total_reward,
            "survival_rate": red_alive / max(self.max_controlled_agents, 1),
        }

        return self._build_observation(), shaped_reward, done, False, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
