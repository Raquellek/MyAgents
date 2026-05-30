"""
env_wrapper.py
==============
Wraps the MAgent2 PettingZoo (AEC) environment into a gym-style
parallel-step interface.

  reset()  → {agent_id: obs_array}
  step({agent_id: action_int})
           → (obs_dict, reward_dict, done_dict, info_dict, positions)
"""

import numpy as np
from typing import Dict, Optional, Tuple


class MAgent2GymWrapper:
    def __init__(
        self,
        env_name:     str            = "battle",
        team:         str            = "red",
        max_steps:    int            = 500,
        map_size:     int            = 45,
        minimap_mode: bool           = False,
        render_mode:  Optional[str]  = None,   # "rgb_array" | None
    ):
        self.team             = team
        self.max_steps        = max_steps
        self._step_count      = 0
        self._initial_team_sz = 0
        self._render_mode     = render_mode

        self._env = self._make_env(env_name, map_size, minimap_mode,
                                   max_steps, render_mode)

        sample = self._first_team_agent(use_possible=True)
        self.obs_shape = self._env.observation_space(sample).shape
        self.n_actions = self._env.action_space(sample).n

    # ------------------------------------------------------------------
    def _make_env(self, name, map_size, minimap_mode, max_steps, render_mode):
        kw = dict(render_mode=render_mode) if render_mode else {}
        if name == "battle":
            from magent2.environments import battle_v4
            return battle_v4.env(
                map_size               = map_size,
                minimap_mode           = minimap_mode,
                step_reward            = -0.05,
                dead_penalty           = -0.1,
                attack_penalty         = -0.1,
                attack_opponent_reward = 0.2,
                max_cycles             = max_steps,
                extra_features         = False,
                **kw,
            )
        elif name == "adversarial_pursuit":
            from magent2.environments import adversarial_pursuit_v4
            return adversarial_pursuit_v4.env(
                map_size=map_size, max_cycles=max_steps, **kw)
        elif name == "gather":
            from magent2.environments import gather_v4
            return gather_v4.env(
                minimap_mode=minimap_mode, max_cycles=max_steps, **kw)
        else:
            raise ValueError(f"Unknown env: {name}. "
                             "Options: battle | adversarial_pursuit | gather")

    # ------------------------------------------------------------------
    def _first_team_agent(self, use_possible=False) -> str:
        pool = self._env.possible_agents if use_possible else self._env.agents
        for a in pool:
            if a.startswith(self.team):
                return a
        return pool[0]

    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, np.ndarray]:
        self._env.reset()
        self._step_count = 0
        obs_dict = self._collect_obs()
        self._initial_team_sz = len(obs_dict)
        return obs_dict

    # ------------------------------------------------------------------
    def _collect_obs(self) -> Dict[str, np.ndarray]:
        out = {}
        for agent in self._env.agents:
            if agent.startswith(self.team):
                obs = self._env.observe(agent)
                if obs is not None:
                    out[agent] = self._preprocess(obs)
        return out

    # ------------------------------------------------------------------
    def step(self, action_dict: Dict[str, int]):
        self._step_count += 1
        reward_dict: Dict[str, float] = {}
        done_dict:   Dict[str, bool]  = {}

        agents_this_round = list(self._env.agents)

        for agent in agents_this_round:
            if agent not in self._env.agents:
                continue

            obs, reward, term, trunc, info = self._env.last()
            done = term or trunc

            if done:
                self._env.step(None)
            elif agent.startswith(self.team):
                act = int(action_dict.get(
                    agent, self._env.action_space(agent).sample()))
                self._env.step(act)
            else:
                self._env.step(self._env.action_space(agent).sample())

            if agent.startswith(self.team):
                reward_dict[agent] = float(reward)
                done_dict[agent]   = done

        obs_dict  = self._collect_obs()
        positions = self.get_positions()
        return obs_dict, reward_dict, done_dict, {}, positions

    # ------------------------------------------------------------------
    def render(self) -> Optional[np.ndarray]:
        """Return an RGB frame (H, W, 3) uint8, or None if not available."""
        try:
            frame = self._env.render()
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                return frame
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def get_positions(self) -> Dict[str, Tuple[int, int]]:
        positions: Dict[str, Tuple[int, int]] = {}
        try:
            inner   = self._env.unwrapped
            handles = inner.get_handles()
            t_idx   = 0 if self.team == "red" else 1
            if t_idx < len(handles):
                pos   = inner.get_pos(handles[t_idx])
                names = [a for a in self._env.agents if a.startswith(self.team)]
                for i, name in enumerate(names):
                    if i < len(pos):
                        positions[name] = (int(pos[i][0]), int(pos[i][1]))
        except Exception:
            for a in self._env.agents:
                if a.startswith(self.team):
                    positions[a] = (0, 0)
        return positions

    # ------------------------------------------------------------------
    def get_win_rate(self) -> float:
        alive = sum(1 for a in self._env.agents if a.startswith(self.team))
        if self._initial_team_sz == 0:
            return 0.0
        return alive / self._initial_team_sz

    # ------------------------------------------------------------------
    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32)
        if obs.max() > 1.0:
            obs = obs / 255.0
        return obs.flatten()

    # ------------------------------------------------------------------
    def close(self):
        self._env.close()