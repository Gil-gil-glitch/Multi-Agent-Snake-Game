import heapq
import random
from config import *

class TacticalBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_action(self, bot_snake, game_state):
        head = bot_snake.body[0]
        map_obj = game_state.map
        snakes = game_state.snakes

        # 1. CHECK FOR INSTANT LAVA-PUSH KILLS (Combat Check)
        opposite_dir = (-bot_snake.direction[0], -bot_snake.direction[1])
        for move in ALL_DIRECTIONS:
            if move == opposite_dir:
                continue
            target_pos = (head[0] + move[0], head[1] + move[1])
            for s in snakes:
                if s.alive and s.team_id != bot_snake.team_id and target_pos in s.body:
                    pushed_pos = (target_pos[0] + move[0], target_pos[1] + move[1])
                    if map_obj.get_tile(pushed_pos) in (TILE_LAVA, TILE_WALL):
                        return move  # TAKE THE KILL IMMEDIATELY!

        # 2. A* PATHFINDING TO NEAREST FOOD
        if map_obj.foods:
            obstacles = set()
            for s in snakes:
                if s.alive:
                    # Treat all snake bodies (except current tail) as impassable
                    obstacles.update(s.body[:-1])

            # Find closest food using A* search
            path = self._a_star_to_nearest_food(head, map_obj, obstacles)
            if path and len(path) > 1:
                next_cell = path[1]
                req_dir = (next_cell[0] - head[0], next_cell[1] - head[1])
                if req_dir in ALL_DIRECTIONS and req_dir != opposite_dir:
                    return req_dir

        # 3. FALLBACK SAFETY MECHANISM (If no food path exists)
        # Pick any valid direction that doesn't kill the snake
        for move in ALL_DIRECTIONS:
            if move == opposite_dir:
                continue
            nr, nc = head[0] + move[0], head[1] + move[1]
            tile = map_obj.get_tile((nr, nc))
            
            # Check collision with snakes
            is_blocked = any((nr, nc) in s.body for s in snakes if s.alive)
            if tile not in (TILE_LAVA, TILE_WALL) and not is_blocked:
                return move

        return bot_snake.direction  # Default hold direction if trapped

    def _a_star_to_nearest_food(self, start, map_obj, obstacles):
        """Runs A* search to find the shortest valid route to any food item on the 100x100 grid."""
        foods = map_obj.foods
        if not foods:
            return None

        # Sort food targets by taxicab distance to start search on closest candidate
        sorted_foods = sorted(foods, key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1]))[:5]

        best_path = None
        shortest_len = float('inf')

        for target in sorted_foods:
            # Queue stores: (f_score, g_score, current_pos, path)
            open_set = []
            heapq.heappush(open_set, (0, 0, start, [start]))
            visited = set([start])

            while open_set:
                f, g, current, path = heapq.heappop(open_set)

                if current == target:
                    if len(path) < shortest_len:
                        shortest_len = len(path)
                        best_path = path
                    break

                if len(path) > 40:  # Search depth limit for performance on 100x100
                    continue

                for dr, dc in ALL_DIRECTIONS:
                    neighbor = (current[0] + dr, current[1] + dc)
                    
                    if neighbor in visited or neighbor in obstacles:
                        continue

                    tile = map_obj.get_tile(neighbor)
                    if tile in (TILE_LAVA, TILE_WALL):
                        continue

                    # Water tile increases traversal movement cost (g_score)
                    step_cost = 2 if tile == TILE_WATER else 1
                    new_g = g + step_cost
                    h = abs(neighbor[0] - target[0]) + abs(neighbor[1] - target[1])
                    new_f = new_g + h

                    visited.add(neighbor)
                    heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))

        return best_path