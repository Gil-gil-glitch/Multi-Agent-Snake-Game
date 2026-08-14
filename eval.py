import torch
import numpy as np
import pygame
from pettingzoo_env import SnakeArenaParallelEnv
from train import SnakeQNetwork

def watch_trained_agents():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Visual PettingZoo Environment
    env = SnakeArenaParallelEnv(render_mode="human")
    
    # 2. Load Trained PyTorch Weights onto GPU
    q_net = SnakeQNetwork().to(device)
    try:
        q_net.load_state_dict(torch.load("snake_qnet.pth", map_location=device))
        q_net.eval()
        print("Successfully loaded trained weights from snake_qnet.pth!")
    except FileNotFoundError:
        print("No saved checkpoint found! Run 'python train.py' first to train and save the model.")
        return

    obs, _ = env.reset()

    running = True
    while env.agents and running:
        actions = {}
        
        # Select best action from Neural Network (Exploration Epsilon = 0)
        for agent in env.agents:
            state_t = torch.FloatTensor(obs[agent]).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = q_net(state_t)
            actions[agent] = q_vals.argmax(dim=1).item()

        obs, rewards, terminations, truncations, _ = env.step(actions)
        env.render()

        # Allow user to close the Pygame window with 'X'
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    env.close()

if __name__ == "__main__":
    watch_trained_agents()