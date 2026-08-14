import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    """Custom Gymnasium Environment for a 2D Grid Maze."""
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self):
        super().__init__()
        
        # 0 = open, 1 = wall, 2 = goal
        self.maze = np.array([
            [0,1,0,1,0,0,0,1,0,0,1,0,0,1,0],
            [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
            [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
            [1,0,1,1,0,1,1,1,0,1,0,1,1,1,0],
            [0,0,1,0,0,0,0,0,1,0,0,0,0,1,0],
            [1,0,1,0,1,0,1,0,0,0,1,0,1,0,0],
            [0,0,0,0,1,0,0,1,0,0,1,0,0,0,0],
            [1,0,1,0,0,1,0,1,0,1,1,1,0,1,0],
            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,1,1,1,1,0,1,1,1,0,1,1,0,1],
            [0,0,0,0,0,0,0,1,0,0,0,0,1,0,0],
            [1,1,0,1,1,1,0,1,0,1,1,0,1,1,0],
            [0,0,0,0,0,0,0,1,0,0,1,0,0,0,0],
            [1,0,1,1,1,1,0,1,1,0,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,2],
        ])

        self.height, self.width = self.maze.shape
        self.start = (0, 0)
        self.goal = (14, 14)
        self.current_pos = self.start

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        
        # State space: Flattened 1D index representing grid cells (0 to height*width - 1)
        self.observation_space = spaces.Discrete(self.height * self.width)

        self._moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R

    def _pos_to_state(self, pos):
        return pos[0] * self.width + pos[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = self.start
        return self._pos_to_state(self.current_pos), {}

    def step(self, action):
        dr, dc = self._moves[action]
        r, c = self.current_pos
        new_r, new_c = r + dr, c + dc

        # Check bounds and wall collisions
        if (0 <= new_r < self.height and 
            0 <= new_c < self.width and 
            self.maze[new_r, new_c] != 1):
            next_pos = (new_r, new_c)
            hit_wall = False
        else:
            next_pos = self.current_pos
            hit_wall = True

        self.current_pos = next_pos
        terminated = (self.maze[next_pos[0], next_pos[1]] == 2)
        
        # Reward Logic
        if terminated:
            reward = 100.0
        elif hit_wall:
            reward = -5.0
        else:
            reward = -0.1

        return self._pos_to_state(self.current_pos), reward, terminated, False, {}

# --- Example Q-Learning Loop using the Gymnasium API ---
if __name__ == "__main__":
    env = MazeEnv()
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    episodes = 500
    lr = 0.2
    gamma = 0.9
    epsilon = 1.0
    min_epsilon = 0.01
    decay = 0.995

    print("Training Q-Learning agent on Gymnasium Environment...")
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-table update
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state, best_next_action]
            q_table[state, action] += lr * (td_target - q_table[state, action])

            state = next_state

        epsilon = max(min_epsilon, epsilon * decay)

    print("Training completed successfully!")