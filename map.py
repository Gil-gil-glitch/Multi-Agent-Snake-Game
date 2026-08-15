import numpy as np
import random
from config import *

class ArenaMap:
    def __init__(self):
        # Initialize grid using config GRID_SIZE
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.foods = set()
        self._generate_terrain()

    def _generate_terrain(self):
        # Generate walls, lava, water within GRID_SIZE bounds
        # Example terrain placement using relative coordinates:
        pass  # Keep your existing terrain logic, ensuring ranges use GRID_SIZE

    def get_tile(self, pos):
        r, c = pos
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return self.grid[r, c]
        return TILE_WALL  # Treat out-of-bounds as wall collision

    def spawn_food(self, occupied_positions, target_count=5):
        attempts = 0
        max_attempts = 100

        while len(self.foods) < target_count and attempts < max_attempts:
            attempts += 1
            r = random.randint(0, GRID_SIZE - 1)
            c = random.randint(0, GRID_SIZE - 1)

            # Avoid spawning on obstacles, existing food, or snakes
            if (r, c) not in occupied_positions and (r, c) not in self.foods:
                tile = self.grid[r, c]
                if tile in (TILE_GRASS, TILE_FARM):
                    self.foods.add((r, c))