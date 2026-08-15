import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv
from functools import lru_cache

from config import *
from game import GameEngine
from snake import Snake

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
        
        # Disable heuristics so RL policy controls all snakes
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

            self.hunger_counters[agent] += 1
            if self.hunger_counters[agent] >= 100:  # Starved to death
                self.terminations[agent] = True
                
            player_id = self.agent_to_id[agent]
            snake = self.game.snakes[player_id - 1]
            team_id = snake.team_id

            # Base Reward: Team Score Delta (Food + Kill Bonuses)
            reward = (self.game.team_scores[team_id] - prev_scores[team_id]) * 0.1

            # Individual Reward Shaping
            if prev_alive[player_id] and not snake.alive:
                reward -= 5.0  # Death Penalty for dying in Lava/Wall
            else:
                reward -= 0.01  # Small step penalty to encourage fast food hunting

            rewards[agent] = float(reward)

            # Termination flag when snake dies
            is_dead = not snake.alive
            terminations[agent] = is_dead
            truncations[agent] = False
            infos[agent] = {"team_score": self.game.team_scores[team_id]}

        # Filter out agents that terminated
        self.agents = [agent for agent in self.agents if not terminations[agent]]
        observations = {agent: self._get_observation(agent) for agent in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    def _get_observation(self, agent):
        player_id = self.agent_to_id[agent]
        current_snake = self.game.snakes[player_id - 1]
        
        obs = np.zeros((5, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Channel 4: Terrain Map (Walls, Lava, Water)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                tile = self.game.map.grid[r, c]
                if tile in (TILE_LAVA, TILE_WALL):
                    obs[4, r, c] = 1.0
                elif tile == TILE_WATER:
                    obs[4, r, c] = 0.5

        # Channel 3: Food
        for r, c in self.game.map.foods:
            obs[3, r, c] = 1.0

        # Channels 0, 1, 2: Snake Bodies
        for s in self.game.snakes:
            if not s.alive:
                continue
            
            # Determine relative channel
            if s.player_id == player_id:
                ch = 0  # Self
            elif s.team_id == current_snake.team_id:
                ch = 1  # Teammate
            else:
                ch = 2  # Enemy

            for idx, (r, c) in enumerate(s.body):
                obs[ch, r, c] = 1.0 if idx == 0 else 0.5

        return obs

    def render(self):
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        import pygame
        pygame.quit()