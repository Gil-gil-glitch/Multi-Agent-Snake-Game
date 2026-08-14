import numpy as np
import random
from config import *

class ArenaMap:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.foods = set()
        self._generate_map()

    def _generate_map(self):
        """
        Method that generates the different parts of the game map.
        """

        # Outer Boundary Walls
        self.grid[0, :] = TILE_WALL
        self.grid[-1, :] = TILE_WALL
        self.grid[:, 0] = TILE_WALL
        self.grid[:, -1] = TILE_WALL

        # Central Farm Zone (High food spawn area: 30x30 in center)
        farm_start, farm_end = 35, 65
        self.grid[farm_start:farm_end, farm_start:farm_end] = TILE_FARM

        # Water Lakes (Slowing hazards)
        self.grid[15:30, 15:35] = TILE_WATER
        self.grid[70:85, 65:85] = TILE_WATER

        # ava Pools & Choke Points (Deadly hazards)
        self.grid[45:55, 10:25] = TILE_LAVA
        self.grid[45:55, 75:90] = TILE_LAVA

    def spawn_food(self, snake_bodies, target_count=30):
        """
        Spawns food across the map, with 70% preference toward Farm zones.
        """

        while len(self.foods) < target_count:
            # Prefer spawning in Farm Zone
            if random.random() < 0.70:
                r = random.randint(35, 64)
                c = random.randint(35, 64)
                
            else:
                r = random.randint(1, self.grid_size - 2)
                c = random.randint(1, self.grid_size - 2)

            tile = self.grid[r, c]
            if tile not in (TILE_LAVA, TILE_WALL) and (r, c) not in snake_bodies and (r, c) not in self.foods:
                self.foods.add((r, c))

    def get_tile(self, pos):
        r, c = pos
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            return self.grid[r, c]
        return TILE_WALL