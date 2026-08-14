import random
from config import *

class TacticalBot:
    def __init__(self, snake_id):
        self.snake_id = snake_id

    def choose_action(self, bot_snake, game_state):
        head = bot_snake.body[0]
        map_obj = game_state.map
        snakes = game_state.snakes

        best_score = -float('inf')
        best_dir = bot_snake.direction

        opposite_dir = (-bot_snake.direction[0], -bot_snake.direction[1])

        for move in ALL_DIRECTIONS:
            # Disallow 180-degree reverse
            if move == opposite_dir:
                continue

            nr, nc = head[0] + move[0], head[1] + move[1]
            target_pos = (nr, nc)
            tile = map_obj.get_tile(target_pos)

            score = 0

            # 1. FATAL HAZARD AVOIDANCE
            if tile in (TILE_LAVA, TILE_WALL):
                score -= 10000

            # Self / Teammate / Enemy Collision Checks
            collided_snake = None
            for s in snakes:
                if s.alive and target_pos in s.body:
                    collided_snake = s
                    break

            if collided_snake:
                if collided_snake.team_id == bot_snake.team_id:
                    # Friendly Teammate -> Slight penalty to avoid cluttering
                    score -= 30
                else:
                    # ENEMY SNAKE FOUND -> Check if we can push them into Lava!
                    pushed_pos = (target_pos[0] + move[0], target_pos[1] + move[1])
                    pushed_tile = map_obj.get_tile(pushed_pos)
                    if pushed_tile in (TILE_LAVA, TILE_WALL):
                        score += 5000  # HUGE BONUS: Environmental Kill Opportunity!
                    else:
                        score -= 50  # Risk pushing enemy onto safe ground

            # 2. TERRAIN PREFERENCES
            if tile == TILE_WATER:
                score -= 15  # Water slows us down
            elif tile == TILE_FARM:
                score += 10   # Farm zone is high value

            # 3. FOOD HUNTING HEURISTIC
            if target_pos in map_obj.foods:
                score += 150
            else:
                # Calculate distance to nearest food
                if map_obj.foods:
                    min_dist = min(abs(nr - fr) + abs(nc - fc) for fr, fc in map_obj.foods)
                    score += (200 - min_dist * 2)

            # Tie-breaker randomness
            score += random.uniform(0, 1)

            if score > best_score:
                best_score = score
                best_dir = move

        return best_dir