import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Maze Definition ---
# 0 = open, 1 = wall, 2 = goal
maze = [
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
]

MAZE_HEIGHT = len(maze)
MAZE_WIDTH = len(maze[0])
START = (0, 0)
GOAL = (14, 14)
MOVES = ['U', 'D', 'L', 'R']
MOVE_TO_IDX = {move: i for i, move in enumerate(MOVES)}

# --- RL Hyperparameters ---
EPISODES = 1000          # Number of training episodes
MAX_STEPS_PER_EPISODE = MAZE_HEIGHT * MAZE_WIDTH * 2 # Max steps before reset
LEARNING_RATE = 0.2      # Alpha (rate at which the Q-values are updated)
DISCOUNT_FACTOR = 0.9    # Gamma (importance of future rewards)
EXPLORATION_RATE = 1.0   # Epsilon (chance of taking a random action)
MAX_EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY_RATE = 0.001


def state_to_index(pos):
    """Converts (x, y) position to a single state index."""
    return pos[0] * MAZE_WIDTH + pos[1]

def get_reward(pos, new_pos):
    """Defines the immediate reward for the transition. There is a small penalty for each move, a big reward for reaching the goal, and a penalty for hitting walls."""
    reward = -0.1
    
    if maze[new_pos[0]][new_pos[1]] == 2:
        reward = 100
    
    elif pos == new_pos:
        reward = -5 

    return reward

def move(pos, direction_idx):
    """Move one step; returns new position and success status. Checks for walls and boundaries."""
    x, y = pos
    direction = MOVES[direction_idx]
    new_x, new_y = x, y

    if direction == 'U': new_x -= 1
    elif direction == 'D': new_x += 1
    elif direction == 'L': new_y -= 1
    elif direction == 'R': new_y += 1
    
    if (new_x < 0 or new_y < 0 or new_x >= MAZE_HEIGHT or 
        new_y >= MAZE_WIDTH or maze[new_x][new_y] == 1):
        return pos, False 
    
    return (new_x, new_y), True 

def choose_action(state_idx, current_epsilon):
    """Epsilon-greedy strategy: explore or exploit. """
    if random.random() < current_epsilon:
        return random.randint(0, len(MOVES) - 1) # Explore
    else:
        # Exploit
        return np.argmax(Q_table[state_idx, :])

# --- Q-Table and Agent Setup ---
# Q-Table size: (Number of states) x (Number of actions)
Q_table = np.zeros((MAZE_HEIGHT * MAZE_WIDTH, len(MOVES)))
current_epsilon = EXPLORATION_RATE
episode = 0
step_count = 0
current_pos = START

# --- Visualization Setup ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("Q-Learning Maze Solver (Training Agent)")
ax.imshow(maze, cmap='binary', origin='upper')
ax.set_xticks(range(MAZE_WIDTH))
ax.set_yticks(range(MAZE_HEIGHT))
ax.grid(True, color='gray', linewidth=0.5)

agent, = ax.plot([], [], 'ro', markersize=8)
goal_marker, = ax.plot(GOAL[1], GOAL[0], 'go', markersize=12)

# --- Q-Learning Update Loop (for Animation) ---

def update_q_learning(frame):
    global current_pos, current_epsilon, episode, step_count

    if episode >= EPISODES:
        print(f"Q-Learning Training Finished after {EPISODES} episodes.")
        anim.event_source.stop()
        return agent,

    state_idx = state_to_index(current_pos)
    action_idx = choose_action(state_idx, current_epsilon)
    
    new_pos, success = move(current_pos, action_idx)
    new_state_idx = state_to_index(new_pos)
    reward = get_reward(current_pos, new_pos)

    old_q_value = Q_table[state_idx, action_idx]
    future_max_q = np.max(Q_table[new_state_idx, :])
    
    new_q_value = old_q_value + LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * future_max_q - old_q_value
    )
    Q_table[state_idx, action_idx] = new_q_value
    
    current_pos = new_pos
    step_count += 1
    
    agent.set_data([current_pos[1]], [current_pos[0]])
    ax.set_title(f"Episode {episode} | Step {step_count} | $\epsilon$: {current_epsilon:.3f}")
    
    # Reset episode
    if maze[current_pos[0]][current_pos[1]] == 2 or step_count >= MAX_STEPS_PER_EPISODE:
        
        goal_reached = maze[current_pos[0]][current_pos[1]] == 2
        
        print(f"\n--- Episode {episode} Completed ---")
        print(f"Goal Status: {'REACHED' if goal_reached else 'FAILED'}")
        print(f"Steps Taken: {step_count}")
        print(f"Final $\epsilon$: {current_epsilon:.4f}")

        current_epsilon = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY_RATE * episode)
        
        episode += 1
        step_count = 0
        current_pos = START # Reset agent to start

    return agent,

anim = animation.FuncAnimation(
    fig, 
    update_q_learning, 
    frames=EPISODES * MAX_STEPS_PER_EPISODE,
    interval=10, 
    blit=True, 
    repeat=False
)

plt.show()
plt.close()

def get_optimal_path(q_table):
    path = [START]
    current = START
    for _ in range(MAZE_HEIGHT * MAZE_WIDTH): 
        if maze[current[0]][current[1]] == 2:
            break
        state_idx = state_to_index(current)
        action_idx = np.argmax(q_table[state_idx, :])
        current, _ = move(current, action_idx)
        path.append(current)
    return path

if episode >= EPISODES:
    optimal_path = get_optimal_path(Q_table)
    print("\nOptimal Path (Post-Training):")
    print(optimal_path)
