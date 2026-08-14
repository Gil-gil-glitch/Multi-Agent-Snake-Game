import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from pettingzoo_env import SnakeArenaParallelEnv

# Convolutional Q-Network for Grid Observations
class SnakeQNetwork(nn.Module):
    def __init__(self, in_channels=5, num_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2), # -> 16x50x50
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),          # -> 32x25x25
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # -> 64x13x13
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 13 * 13, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# Multi-Agent Training Loop
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeArenaParallelEnv(render_mode=None)
    
    # Shared Policy Network for all snakes (Self-Play)
    q_net = SnakeQNetwork().to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    replay_buffer = deque(maxlen=50000)
    
    epsilon = 1.0
    batch_size = 32
    gamma = 0.99

    print(f"Training Multi-Agent Snake Arena on {device}...")

    for episode in range(1000):
        obs, _ = env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}
        
        while env.agents:
            actions = {}
            
            # Select Epsilon-Greedy Action for each active agent
            for agent in env.agents:
                if random.random() < epsilon:
                    actions[agent] = env.action_space(agent).sample()
                else:
                    state_t = torch.FloatTensor(obs[agent]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_vals = q_net(state_t)
                    actions[agent] = q_vals.argmax(dim=1).item()

            # Environment Step
            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            # Store transitions in Replay Buffer
            for agent in actions.keys():
                replay_buffer.append((
                    obs[agent],
                    actions[agent],
                    rewards[agent],
                    next_obs[agent],
                    terminations[agent]
                ))
                total_rewards[agent] += rewards[agent]

            obs = next_obs

            # Optimize PyTorch Network
            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                s_b, a_b, r_b, ns_b, d_b = zip(*batch)

                s_tensor = torch.FloatTensor(np.array(s_b)).to(device)
                a_tensor = torch.LongTensor(a_b).unsqueeze(1).to(device)
                r_tensor = torch.FloatTensor(r_b).unsqueeze(1).to(device)
                ns_tensor = torch.FloatTensor(np.array(ns_b)).to(device)
                d_tensor = torch.FloatTensor(d_b).unsqueeze(1).to(device)

                curr_q = q_net(s_tensor).gather(1, a_tensor)
                with torch.no_grad():
                    max_next_q = q_net(ns_tensor).max(dim=1, keepdim=True)[0]
                    target_q = r_tensor + (1 - d_tensor) * gamma * max_next_q

                loss = nn.MSELoss()(curr_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(0.05, epsilon * 0.995)

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1:4d} | Cyan Avg Reward: {total_rewards['snake_0']:.2f} | Epsilon: {epsilon:.2f}")

    env.close()