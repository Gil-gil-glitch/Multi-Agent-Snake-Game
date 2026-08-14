#
##  game.py
#
#   This is the main file of the Multi-Agent Snake game that holds the 
#   game logic along with its environment.
#
#

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class CompetitiveSnakeEnv(gym.Env):
    """
    2-Player Competitive Snake Environment with Pygame rendering.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=20, cell_size=25, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode
        self.window_size = (grid_size * cell_size, grid_size * cell_size)

        # 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Tuple([
            spaces.Discrete(4),
            spaces.Discrete(4)
        ])

        # Multi-channel Observation Space for CNN / RL Agents:
        # Channel 0: Self, Channel 1: Opponent, Channel 2: Food
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(3, self.grid_size, self.grid_size), 
            dtype=np.float32
        )

        self._moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R

        # Pygame setup
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initial Snake Positions (Spawning on opposite corners)
        self.snake_1 = [(3, 3), (3, 2), (3, 1)]
        self.dir_1 = 3  # Facing Right

        self.snake_2 = [(self.grid_size - 4, self.grid_size - 4), 
                        (self.grid_size - 4, self.grid_size - 3), 
                        (self.grid_size - 4, self.grid_size - 2)]
        self.dir_2 = 2  # Facing Left

        self.scores = [0, 0]
        self._spawn_food()

        obs_1 = self._get_obs(player=1)
        obs_2 = self._get_obs(player=2)

        return (obs_1, obs_2), {}

    def _spawn_food(self):
        """
        Spawns food in an empty cell not occupied by either snake.
        """
        occupied = set(self.snake_1) | set(self.snake_2)
        all_cells = {(r, c) for r in range(self.grid_size) for c in range(self.grid_size)}
        free_cells = list(all_cells - occupied)
        
        if free_cells:
            self.food = random.choice(free_cells)
        else:
            self.food = (-1, -1)

    def _get_obs(self, player=1):
        """Constructs a 3-channel grid tensor for the specified player."""
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        
        self_body = self.snake_1 if player == 1 else self.snake_2
        opp_body = self.snake_2 if player == 1 else self.snake_1

        # Channel 0: Self
        for r, c in self_body:
            obs[0, r, c] = 1.0
            
        # Channel 1: Opponent
        for r, c in opp_body:
            obs[1, r, c] = 1.0

        # Channel 2: Food
        if self.food != (-1, -1):
            obs[2, self.food[0], self.food[1]] = 1.0

        return obs

    def step(self, actions):
        """Executes simultaneous moves for both snakes."""
        a1, a2 = actions
        
        # Prevent instant 180-degree self-collisions
        if (a1 == 0 and self.dir_1 != 1) or (a1 == 1 and self.dir_1 != 0) or \
           (a1 == 2 and self.dir_1 != 3) or (a1 == 3 and self.dir_1 != 2):
            self.dir_1 = a1
            
        if (a2 == 0 and self.dir_2 != 1) or (a2 == 1 and self.dir_2 != 0) or \
           (a2 == 2 and self.dir_2 != 3) or (a2 == 3 and self.dir_2 != 2):
            self.dir_2 = a2

        # Calculate new head positions
        dr1, dc1 = self._moves[self.dir_1]
        dr2, dc2 = self._moves[self.dir_2]
        
        new_head_1 = (self.snake_1[0][0] + dr1, self.snake_1[0][1] + dc1)
        new_head_2 = (self.snake_2[0][0] + dr2, self.snake_2[0][1] + dc2)

        # Collision Flags
        dead_1, dead_2 = False, False
        r1, r2 = -0.01, -0.01  # Small step penalty

        # Check Head-on Collision
        if new_head_1 == new_head_2:
            dead_1, dead_2 = True, True
        else:
            # Check Snake 1 Collisions
            if (not (0 <= new_head_1[0] < self.grid_size and 0 <= new_head_1[1] < self.grid_size) or
                new_head_1 in self.snake_1[:-1] or new_head_1 in self.snake_2):
                dead_1 = True

            # Check Snake 2 Collisions
            if (not (0 <= new_head_2[0] < self.grid_size and 0 <= new_head_2[1] < self.grid_size) or
                new_head_2 in self.snake_2[:-1] or new_head_2 in self.snake_1):
                dead_2 = True

        # Resolve Rewards and Tail Movement
        if dead_1: r1 = -10.0
        if dead_2: r2 = -10.0

        # Update Snake 1 Position
        if not dead_1:
            self.snake_1.insert(0, new_head_1)
            if new_head_1 == self.food:
                r1 = 10.0
                self.scores[0] += 1
                self._spawn_food()
            else:
                self.snake_1.pop()

        # Update Snake 2 Position
        if not dead_2:
            self.snake_2.insert(0, new_head_2)
            if new_head_2 == self.food:
                r2 = 10.0
                self.scores[1] += 1
                self._spawn_food()
            else:
                self.snake_2.pop()

        terminated = dead_1 or dead_2
        
        obs_1 = self._get_obs(player=1)
        obs_2 = self._get_obs(player=2)

        return (obs_1, obs_2), (r1, r2), terminated, False, {}

    def render(self):
        """Renders the game state using Pygame."""
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("2-Player Competitive Snake RL")
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((20, 20, 25))  # Dark background

        # Draw Grid
        for x in range(0, self.window_size[0], self.cell_size):
            pygame.draw.line(canvas, (40, 40, 50), (x, 0), (x, self.window_size[1]))
        for y in range(0, self.window_size[1], self.cell_size):
            pygame.draw.line(canvas, (40, 40, 50), (0, y), (self.window_size[0], y))

        # Draw Food (Gold)
        if self.food != (-1, -1):
            fr, fc = self.food
            rect = pygame.Rect(fc * self.cell_size, fr * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(canvas, (255, 215, 0), rect)

        # Draw Snake 1 (Cyan)
        for i, (r, c) in enumerate(self.snake_1):
            color = (0, 255, 200) if i == 0 else (0, 180, 140)
            rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(canvas, color, rect)

        # Draw Snake 2 (Magenta)
        for i, (r, c) in enumerate(self.snake_2):
            color = (255, 0, 128) if i == 0 else (180, 0, 90)
            rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(canvas, color, rect)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


# --- Run Game Loop with Heuristic / Random Agents ---
if __name__ == "__main__":
    env = CompetitiveSnakeEnv(render_mode="human")
    (obs1, obs2), _ = env.reset()
    done = False

    def simple_heuristic(snake, food):
        """Basic greedy heuristic to steer head toward food."""
        head_r, head_c = snake[0]
        food_r, food_c = food
        if food_r < head_r: return 0  # Up
        if food_r > head_r: return 1  # Down
        if food_c < head_c: return 2  # Left
        return 3                      # Right

    print("Running 2-Snake Competitive Environment...")
    while not done:
        # Snake 1 uses greedy heuristic, Snake 2 takes random moves
        a1 = simple_heuristic(env.snake_1, env.food)
        a2 = env.action_space[1].sample()

        (obs1, obs2), (r1, r2), terminated, truncated, _ = env.step([a1, a2])
        done = terminated or truncated

        env.render()

    print(f"Game Over! Final Scores -> Snake 1 (Cyan): {env.scores[0]} | Snake 2 (Magenta): {env.scores[1]}")
    env.close()