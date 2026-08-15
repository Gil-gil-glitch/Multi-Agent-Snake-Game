import numpy as np
from config import GRID_SIZE, TILE_LAVA, TILE_WALL, TILE_WATER


def build_observation(game_map, snakes, current_snake):
    """Builds the 5xGRID_SIZExGRID_SIZE tensor a snake "sees":
       Ch0 = self body, Ch1 = teammate bodies, Ch2 = enemy bodies,
       Ch3 = food, Ch4 = terrain (1.0 lava/wall, 0.5 water).
       Head cells are 1.0, body cells are 0.5, matching AnmitsuSnakeBot's
       expectations (and reused as-is by pettingzoo_env for RL observations).
    """
    obs = np.zeros((5, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            tile = game_map.grid[r, c]
            if tile in (TILE_LAVA, TILE_WALL):
                obs[4, r, c] = 1.0
            elif tile == TILE_WATER:
                obs[4, r, c] = 0.5

    for r, c in game_map.foods:
        obs[3, r, c] = 1.0

    for s in snakes:
        if not s.alive:
            continue
        if s is current_snake:
            ch = 0
        elif s.team_id == current_snake.team_id:
            ch = 1
        else:
            ch = 2
        for idx, (r, c) in enumerate(s.body):
            obs[ch, r, c] = 1.0 if idx == 0 else 0.5

    return obs