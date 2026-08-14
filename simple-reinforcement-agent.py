import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    """Custom Gymnasium Environment for a 2D Grid Maze with Matplotlib Rendering."""
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
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
        self.observation_space = spaces.Discrete(self.height * self.width)
        self._moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R

        # Matplotlib visualization attributes
        self.fig = None
        self.ax = None
        self.agent_marker = None

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

        # Bounds and wall collisions check
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
        
        # Reward logic
        if terminated:
            reward = 100.0
        elif hit_wall:
            reward = -5.0
        else:
            reward = -0.1

        return self._pos_to_state(self.current_pos), reward, terminated, False, {}

    def render(self):
        """Renders the environment state using Matplotlib."""
        if self.render_mode != "human":
            return

        # Initialize plot window on first call
        if self.fig is None or self.ax is None:
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.ax.set_title("Gymnasium Maze Environment")
            self.ax.imshow(self.maze, cmap='binary', origin='upper')
            self.ax.set_xticks(range(self.width))
            self.ax.set_yticks(range(self.height))
            self.ax.grid(True, color='gray', linewidth=0.5)

            # Static Goal Marker (Green)
            self.ax.plot(self.goal[1], self.goal[0], 'go', markersize=12, label="Goal")
            
            # Dynamic Agent Marker (Red)
            self.agent_marker, = self.ax.plot(
                self.current_pos[1], self.current_pos[0], 'ro', markersize=8, label="Agent"
            )
            self.ax.legend(loc="upper right")
        else:
            # Update only the agent marker position for smooth rendering
            self.agent_marker.set_data([self.current_pos[1]], [self.current_pos[0]])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)  # Frame delay (seconds)

    def close(self):
        """Closes the Matplotlib plot window cleanly."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# --- Training & Live Visualization Script ---
if __name__ == "__main__":
    # 1. Train Offline (Fast, no rendering)
    train_env = MazeEnv()
    q_table = np.zeros((train_env.observation_space.n, train_env.action_space.n))

    episodes = 500
    lr = 0.2
    gamma = 0.9
    epsilon = 1.0
    min_epsilon = 0.01
    decay = 0.995

    print("Training Q-Learning agent...")
    for ep in range(episodes):
        state, _ = train_env.reset()
        done = False
        
        while not done:
            if np.random.rand() < epsilon:
                action = train_env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated

            # Q-update
            q_table[state, action] += lr * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )
            state = next_state

        epsilon = max(min_epsilon, epsilon * decay)

    print("Training finished!\n")

    # 2. Watch Trained Agent Solve Maze
    render_env = MazeEnv(render_mode="human")
    state, _ = render_env.reset()
    done = False
    step_count = 0

    print("Watching trained agent solve the maze...")
    render_env.render()

    while not done:
        # Pure exploitation: take best action according to Q-table
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        step_count += 1
        
        render_env.render()

    print(f"Goal reached in {step_count} steps!")
    plt.show(block=True)  # Keep window open after reaching the goal
    render_env.close()