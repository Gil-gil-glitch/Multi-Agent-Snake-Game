import numpy as np
from collections import deque

class AnmitsuSnakeBot:
    def __init__(self, agent_id="snake_0", max_hunger=100):
        self.agent_id = agent_id
        self.max_hunger = max_hunger
        self.hunger = 0
        self.prev_length = 0

    def get_action(self, obs):
        # Update hunger based on snake length expansion
        current_length = np.sum(obs[0] > 0)
        if self.prev_length > 0 and current_length > self.prev_length:
            self.hunger = 0  # Reset hunger on eating food
        else:
            self.hunger += 1
        self.prev_length = current_length

        weights = self._get_phase_weights()

        best_action = 0
        best_value = float('-inf')

        for action in range(4):
            features = self._extract_features(obs, action)
            # Add uniform noise to break exact move ties
            value = np.dot(weights, features) + np.random.uniform(0, 0.01)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _get_phase_weights(self):
        hunger_ratio = self.hunger / float(self.max_hunger)

        if hunger_ratio < 0.35:
            # PHASE 1: SATIATED -> Maximize space & evasion
            return np.array([1000.0, -800.0, 30.0, 5.0])
        elif hunger_ratio < 0.70:
            # PHASE 2: HUNGRY -> Balanced food seeking
            return np.array([1000.0, -600.0, 15.0, 40.0])
        else:
            # PHASE 3: STARVING -> Aggressive food rush
            return np.array([1000.0, -200.0, 5.0, 100.0])

    def _extract_features(self, obs, action):
        head_pos = self._find_head(obs)
        if head_pos is None:
            return np.zeros(4)

        r, c = head_pos
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr, nc = r + dr, c + dc

        grid_h, grid_w = obs.shape[1], obs.shape[2]

        # F0: Hard Obstacle Collision (Wall, Lava, Snake Bodies)
        if nr < 0 or nr >= grid_h or nc < 0 or nc >= grid_w or self._is_blocked(obs, nr, nc):
            return np.array([-1.0, 0.0, 0.0, 0.0])
        
        f0_safety = 1.0
        f1_head_hazard = 1.0 if self._is_near_opponent_head(obs, nr, nc) else 0.0
        free_space = self._flood_fill_space(obs, nr, nc, max_depth=100)
        f2_space = free_space / 100.0
        food_dist = self._nearest_food_distance(obs, nr, nc)
        f3_food = 1.0 / (food_dist + 1.0) if food_dist < float('inf') else 0.0

        return np.array([f0_safety, f1_head_hazard, f2_space, f3_food])

    def _find_head(self, obs):
        # Head is stored as 1.0 in Channel 0
        coords = np.argwhere(obs[0] == 1.0)
        return tuple(coords[0]) if len(coords) > 0 else None

    def _is_blocked(self, obs, r, c):
        # Channel 0: Self | Channel 1: Teammate | Channel 2: Enemy | Channel 4: Terrain
        self_body = obs[0][r, c] > 0
        teammate_body = obs[1][r, c] > 0
        enemy_body = obs[2][r, c] > 0
        wall_lava = obs[4][r, c] > 0.8  # >0.8 flags Wall/Lava, permits Water (0.5)

        return self_body or teammate_body or enemy_body or wall_lava

    def _is_near_opponent_head(self, obs, r, c):
        # Enemy heads are stored as 1.0 in Channel 2
        enemy_heads = np.argwhere(obs[2] == 1.0)
        if len(enemy_heads) == 0:
            return False

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_r, adj_c = r + dr, c + dc
            if any(np.array_equal([adj_r, adj_c], head) for head in enemy_heads):
                return True
        return False

    def _flood_fill_space(self, obs, start_r, start_c, max_depth=100):
        queue = deque([(start_r, start_c)])
        visited = {(start_r, start_c)}
        grid_h, grid_w = obs.shape[1], obs.shape[2]

        while queue and len(visited) < max_depth:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w and (nr, nc) not in visited:
                    if not self._is_blocked(obs, nr, nc):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return len(visited)

    def _nearest_food_distance(self, obs, start_r, start_c):
        # Food is stored in Channel 3
        food_coords = np.argwhere(obs[3] == 1.0)
        if len(food_coords) == 0:
            return float('inf')
        distances = [abs(start_r - fr) + abs(start_c - fc) for fr, fc in food_coords]
        return min(distances)