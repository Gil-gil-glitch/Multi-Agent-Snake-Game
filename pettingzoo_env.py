import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv
from functools import lru_cache

from config import *
from game import GameEngine
from snake import Snake
from observation import build_observation

class SnakeArenaParallelEnv(ParallelEnv):
    metadata = {
        "name": "snake_arena_2v2_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.game = GameEngine()
        
        # 4 Agents: snake_0 & snake_1 (Cyan Team), snake_2 & snake_3 (Magenta Team)
        self.possible_agents = ["snake_0", "snake_1", "snake_2", "snake_3"]
        self.agent_to_id = {agent: idx + 1 for idx, agent in enumerate(self.possible_agents)}
        self.hunger_counters = {agent: 0 for agent in self.possible_agents}

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        # 5 Channels x 100 Rows x 100 Columns grid tensor
        return Box(low=0.0, high=1.0, shape=(5, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        # 0: Up, 1: Down, 2: Left, 3: Right
        return Discrete(4)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.game.reset()
        self.hunger_counters = {agent: 0 for agent in self.possible_agents}
        
        # Disable heuristics so external policies control all snakes
        for s in self.game.snakes:
            s.is_bot = False

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        # 1. Apply Actions
        action_map = {0: DIR_UP, 1: DIR_DOWN, 2: DIR_LEFT, 3: DIR_RIGHT}
        for agent, act_idx in actions.items():
            player_id = self.agent_to_id[agent]
            snake = self.game.snakes[player_id - 1]
            if snake.alive:
                snake.set_direction(action_map[act_idx])

        # Track state before engine update to calculate RL reward deltas
        prev_scores = self.game.team_scores.copy()
        prev_alive = {s.player_id: s.alive for s in self.game.snakes}

        # 2. Advance Game Engine 1 Tick
        self.game.update()
        if self.render_mode == "human":
            self.game.render()

        # 3. Calculate Rewards & Status
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.possible_agents:
            if agent not in self.agents:
                continue

            player_id = self.agent_to_id[agent]
            snake = self.game.snakes[player_id - 1]
            team_id = snake.team_id

            # Score delta check (Reset hunger on food consumption)
            score_delta = self.game.team_scores[team_id] - prev_scores[team_id]
            if score_delta > 0:
                self.hunger_counters[agent] = 0
            else:
                self.hunger_counters[agent] += 1

            # Starvation condition
            starved = False
            if self.hunger_counters[agent] >= 100:
                starved = True
                snake.alive = False  # Kill snake in game engine

            # Reward Calculation
            reward = score_delta * 0.1

            if (prev_alive[player_id] and not snake.alive) or starved:
                reward -= 5.0  # Death Penalty for Wall, Lava, Collision, or Starvation
            else:
                reward -= 0.01  # Step penalty

            rewards[agent] = float(reward)

            # Termination Status
            is_dead = not snake.alive
            terminations[agent] = is_dead
            truncations[agent] = False
            infos[agent] = {
                "team_score": self.game.team_scores[team_id],
                "hunger": self.hunger_counters[agent]
            }

        # Filter out terminated agents
        self.agents = [agent for agent in self.agents if not terminations[agent]]
        observations = {agent: self._get_observation(agent) for agent in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    def _get_observation(self, agent):
        player_id = self.agent_to_id[agent]
        current_snake = self.game.snakes[player_id - 1]
        return build_observation(self.game.map, self.game.snakes, current_snake)

    def render(self):
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        import pygame
        pygame.quit()