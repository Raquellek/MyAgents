import numpy as np
from gymnasium import spaces
from magent2.environments import battle_v4
from pettingzoo import ParallelEnv

from opponents import RandomOpponent


def unpack_reset(reset_result):
    if isinstance(reset_result, tuple):
        obs, _infos = reset_result
        return obs

    return reset_result


class BattleV4MAPPOParallelEnv(ParallelEnv):
    metadata = {"name": "battle_v4_mappo_like", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        map_size=45,
        max_cycles=500,
        render_mode=None,
        controlled_team="red",
        opponent_pool=None,
        reward_mode="flee",
    ):
        self.env = battle_v4.parallel_env(
            map_size=map_size,
            max_cycles=max_cycles,
            render_mode=render_mode,
        )
        self.map_size = map_size
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.controlled_team = controlled_team
        self.enemy_team = "blue" if controlled_team == "red" else "red"
        self.opponent_pool = opponent_pool or [RandomOpponent()]
        self.current_opponent = self.opponent_pool[0]
        self.reward_mode = reward_mode

        self._obs = unpack_reset(self.env.reset(seed=42))
        self.possible_agents = [
            agent for agent in self.env.agents if agent.startswith(self.controlled_team)
        ]
        self.agents = list(self.possible_agents)

        first_agent = self.possible_agents[0]
        self._observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=self._obs[first_agent].shape,
            dtype=np.float32,
        )
        self._action_space = self.env.action_space(first_agent)

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def controlled_agents(self):
        return [
            agent for agent in self.env.agents
            if agent.startswith(self.controlled_team)
        ]

    def enemy_agents(self):
        return [
            agent for agent in self.env.agents
            if agent.startswith(self.enemy_team)
        ]

    def _team_counts(self):
        red_alive = len([agent for agent in self.env.agents if agent.startswith("red")])
        blue_alive = len([agent for agent in self.env.agents if agent.startswith("blue")])
        return red_alive, blue_alive

    def _shape_reward(
        self,
        agent,
        raw_reward,
        alive_after,
        red_alive,
        blue_alive,
        pre_red_alive,
        pre_blue_alive,
        done,
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

        ally_lost = max(pre_ally_alive - ally_alive, 0)
        enemy_lost = max(pre_enemy_alive - enemy_alive, 0)
        survival_ratio = ally_alive / max(len(self.possible_agents), 1)

        reward = 0.25 * raw_reward
        reward += 0.03 * ally_alive
        reward += 2.0 * enemy_lost

        if alive_after:
            reward += 0.05
        else:
            reward -= 2.0

        if self.reward_mode == "flee":
            reward += 0.5 * survival_ratio
            reward -= 2.0 * ally_lost
        else:
            reward += 0.04 * (ally_alive - enemy_alive)

        if done:
            if ally_alive == 0:
                reward -= 10.0
            elif enemy_alive == 0 and ally_alive > 0:
                reward += 10.0 * survival_ratio
            elif ally_alive > enemy_alive:
                reward += 5.0
            elif ally_alive < enemy_alive:
                reward -= 5.0

        return float(reward)

    def _observations(self):
        return {
            agent: self._obs[agent].astype(np.float32)
            for agent in self.agents
            if agent in self._obs
        }

    def reset(self, seed=None, options=None):
        self._obs = unpack_reset(self.env.reset(seed=seed))
        self.current_opponent = np.random.choice(self.opponent_pool)
        self.agents = self.controlled_agents()
        return self._observations(), {agent: {} for agent in self.agents}

    def step(self, actions):
        env_actions = {}
        pre_agents = list(self.agents)
        pre_red_alive, pre_blue_alive = self._team_counts()

        for agent in self.controlled_agents():
            if agent in self._obs and agent in actions:
                env_actions[agent] = int(actions[agent])

        for agent in self.enemy_agents():
            if agent in self._obs:
                env_actions[agent] = self.current_opponent.act(
                    agent,
                    self._obs[agent],
                    self.env,
                )

        next_obs, raw_rewards, terminations, truncations, _infos = self.env.step(env_actions)
        self._obs = next_obs

        red_alive, blue_alive = self._team_counts()
        all_truncated = len(truncations) > 0 and all(truncations.values())

        if all_truncated and len(self.env.agents) == 0:
            red_alive = pre_red_alive
            blue_alive = pre_blue_alive

        done = (
            red_alive == 0
            or blue_alive == 0
            or len(self.env.agents) == 0
            or (len(terminations) > 0 and all(terminations.values()))
            or all_truncated
        )

        next_controlled = self.controlled_agents()
        rewards = {}
        terminations_out = {}
        truncations_out = {}
        infos = {}

        for agent in pre_agents:
            alive_after = agent in next_controlled and agent in self._obs
            rewards[agent] = self._shape_reward(
                agent=agent,
                raw_reward=raw_rewards.get(agent, 0.0),
                alive_after=alive_after,
                red_alive=red_alive,
                blue_alive=blue_alive,
                pre_red_alive=pre_red_alive,
                pre_blue_alive=pre_blue_alive,
                done=done,
            )
            terminations_out[agent] = done or not alive_after
            truncations_out[agent] = False
            infos[agent] = {
                "red_alive": red_alive,
                "blue_alive": blue_alive,
                "raw_reward": raw_rewards.get(agent, 0.0),
                "survival_rate": red_alive / max(len(self.possible_agents), 1),
            }

        if done:
            self.agents = []
        else:
            self.agents = next_controlled

        return self._observations(), rewards, terminations_out, truncations_out, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
