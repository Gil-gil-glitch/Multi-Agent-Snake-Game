import numpy as np
from collections import deque

class AnmitsuSnakeBot:
    """
    A rule-based heuristic Snake agent inspired by feature-weighted Reversi bots.
    Evaluates moves using collision safety, flood-fill room, food distance, and trap penalty.
    """
    def __init__(self, agent_id="snake_0"):
        self.agent_id = agent_id
        
        # Feature Weights:
        # [0] Survival / Immediate Safety
        # [1] Flood Fill Space (Reachable Tiles)
        # [2] Distance to Nearest Food (Inverted)
        # [3] Trapped / Squeeze Hazard Penalty
        self.weights = np.array([1000.0, 15.0, 25.0, -50.0])

    def get_action(self, obs):
        """
        Evaluates all 4 actions (0: Up, 1: Down, 2: Left, 3: Right) and selects 
        the action yielding the highest heuristic score.
        """
        best_action = 0
        best_value = float('-inf')

        for action in range(4):
            features = self._extract_features(obs, action)
            value = np.dot(self.weights, features)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _extract_features(self, obs, action):
        """
        Extracts f(S') feature vector for taking `action` from current observation.
        """

        # Unpack channels (assuming standard grid representation):
        # channel 0: my snake head & body | channel 1: opponents | channel 2: food | channel 3: walls/lava
        
        head_pos = self._find_head(obs)
        if head_pos is None:
            return np.zeros(len(self.weights))

        r, c = head_pos
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr, nc = r + dr, c + dc

        grid_h, grid_w = obs.shape[1], obs.shape[2]

        # F0: Immediate Collision Safety
        if nr < 0 or nr >= grid_h or nc < 0 or nc >= grid_w:
            return np.array([-1.0, 0.0, 0.0, 1.0])  # Wall crash
        
        is_obstacle = self._is_blocked(obs, nr, nc)
        if is_obstacle:
            return np.array([-1.0, 0.0, 0.0, 1.0])  # Body/Lava crash

        f0_safety = 1.0

        # F1: Flood Fill Available Space (BFS up to 100 tiles)
        free_space = self._flood_fill_space(obs, nr, nc, max_depth=100)
        f1_space = free_space / 100.0

        # F2: Inverted Distance to Closest Food
        food_dist = self._nearest_food_distance(obs, nr, nc)
        f2_food_prox = 1.0 / (food_dist + 1.0) if food_dist < float('inf') else 0.0

        # F3: Trap Hazard (Severe penalty if available space is smaller than body length)
        f3_trap = 1.0 if free_space < 15 else 0.0

        return np.array([f0_safety, f1_space, f2_food_prox, f3_trap])

    def _find_head(self, obs):
        # Locate snake head in channel 0
        coords = np.argwhere(obs[0] == 1)
        return tuple(coords[0]) if len(coords) > 0 else None

    def _is_blocked(self, obs, r, c):
        # Checks if coordinate contains snake bodies or hazards
        my_body = obs[0][r, c] > 0
        opponent_body = obs[1][r, c] > 0 if obs.shape[0] > 1 else False
        hazard_lava = obs[3][r, c] > 0 if obs.shape[0] > 3 else False
        return my_body or opponent_body or hazard_lava

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
        
        # Manhattan distance to nearest food item
        distances = [abs(start_r - fr) + abs(start_c - fc) for fr, fc in food_coords]
        return min(distances)