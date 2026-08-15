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
        # Scatter roughly-circular biome patches across the grid. Patches only
        # overwrite plain grass, so biomes never eat into each other and the
        # counts/radii scale with GRID_SIZE via the config constants.
        self._place_biome_patches(TILE_LAVA, LAVA_PATCHES, *LAVA_RADIUS_RANGE)
        self._place_biome_patches(TILE_WATER, WATER_PATCHES, *WATER_RADIUS_RANGE)
        self._place_biome_patches(TILE_WALL, WALL_PATCHES, *WALL_RADIUS_RANGE)
        self._place_biome_patches(TILE_FARM, FARM_PATCHES, *FARM_RADIUS_RANGE)

    def _place_biome_patches(self, tile_type, num_patches, min_radius, max_radius):
        margin = max(2, max_radius + 1)
        if GRID_SIZE - margin <= margin:
            return  # Grid too small for this patch size, skip safely

        for _ in range(num_patches):
            center_r = random.randint(margin, GRID_SIZE - 1 - margin)
            center_c = random.randint(margin, GRID_SIZE - 1 - margin)
            radius = random.randint(min_radius, max_radius)

            for r in range(center_r - radius, center_r + radius + 1):
                for c in range(center_c - radius, center_c + radius + 1):
                    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                        if (r - center_r) ** 2 + (c - center_c) ** 2 <= radius ** 2:
                            if self.grid[r, c] == TILE_GRASS:
                                self.grid[r, c] = tile_type

    def clear_area(self, center, radius=SPAWN_CLEAR_RADIUS):
        """Removes lethal hazards (lava/wall) from around a point - used to
        guarantee a snake never spawns on top of instant-death terrain."""
        cr, cc = center
        for r in range(cr - radius, cr + radius + 1):
            for c in range(cc - radius, cc + radius + 1):
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    if self.grid[r, c] in (TILE_LAVA, TILE_WALL):
                        self.grid[r, c] = TILE_GRASS

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