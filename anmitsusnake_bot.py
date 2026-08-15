import numpy as np
from collections import deque

class AnmitsuSnakeBot:
    def __init__(self, agent_id="snake_0"):
        self.agent_id = agent_id

        # Weights:
        # [0] Survival / Static Collision (-1000 if wall/body)
        # [1] Opponent Head Danger Zone (Avoid head-on collisions)
        # [2] Flood Fill Space (Reachable area)
        # [3] Food Distance (Inverted)
        self.weights = np.array([1000.0, -800.0, 15.0, 20.0])

    def get_action(self, obs):
        best_action = 0
        best_value = float('-inf')

        # Add small random noise to break exact ties when targeting food
        for action in range(4):
            features = self._extract_features(obs, action)
            value = np.dot(self.weights, features) + np.random.uniform(0, 0.01)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _extract_features(self, obs, action):
        head_pos = self._find_head(obs)
        if head_pos is None:
            return np.zeros(len(self.weights))

        r, c = head_pos
        # Action Map: 0=Up, 1=Down, 2=Left, 3=Right
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr, nc = r + dr, c + dc

        grid_h, grid_w = obs.shape[1], obs.shape[2]

        # F0: Immediate Static Collision (Wall, Body, Lava)
        if nr < 0 or nr >= grid_h or nc < 0 or nc >= grid_w or self._is_blocked(obs, nr, nc):
            return np.array([-1.0, 0.0, 0.0, 0.0])
        
        f0_safety = 1.0

        # F1: Opponent Head Hazard Zone (1 step ahead prediction)
        f1_head_hazard = 1.0 if self._is_near_opponent_head(obs, nr, nc) else 0.0

        # F2: Flood Fill Reachable Space
        free_space = self._flood_fill_space(obs, nr, nc, max_depth=100)
        f2_space = free_space / 100.0

        # F3: Distance to Nearest Food
        food_dist = self._nearest_food_distance(obs, nr, nc)
        f3_food = 1.0 / (food_dist + 1.0) if food_dist < float('inf') else 0.0

        return np.array([f0_safety, f1_head_hazard, f2_space, f3_food])

    def _find_head(self, obs):
        coords = np.argwhere(obs[0] == 1)  # Channel 0 is player head/body
        return tuple(coords[0]) if len(coords) > 0 else None

    def _is_blocked(self, obs, r, c):
        my_body = obs[0][r, c] > 0
        opponent_body = obs[1][r, c] > 0 if obs.shape[0] > 1 else False
        hazard_lava = obs[3][r, c] > 0 if obs.shape[0] > 3 else False
        return my_body or opponent_body or hazard_lava

    def _is_near_opponent_head(self, obs, r, c):
        """Checks if target cell (r, c) is adjacent to any opponent head."""
        if obs.shape[0] <= 1:
            return False
        
        # Channel 1: Opponents
        opp_coords = np.argwhere(obs[1] == 1)
        if len(opp_coords) == 0:
            return False

        # Check orthogonal neighbors of (r, c) for opponent presence
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_r, adj_c = r + dr, c + dc
            if any(np.array_equal([adj_r, adj_c], opp_head) for opp_head in opp_coords):
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
        food_coords = np.argwhere(obs[2] == 1) if obs.shape[0] > 2 else []
        if len(food_coords) == 0:
            return float('inf')
        distances = [abs(start_r - fr) + abs(start_c - fc) for fr, fc in food_coords]
        return min(distances)