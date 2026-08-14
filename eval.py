import torch
import numpy as np
import pygame
from pettingzoo_env import SnakeArenaParallelEnv
from train import SnakeQNetwork

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading trained model on {device}...")

    # Initialize Visual PettingZoo Environment
    env = SnakeArenaParallelEnv(render_mode="human")
    
    # Load Trained Weights
    q_net = SnakeQNetwork().to(device)
    try:
        q_net.load_state_dict(torch.load("snake_qnet.pth", map_location=device))
        q_net.eval()  # Set network to evaluation mode
        print("Model loaded successfully! Starting match...")
    except FileNotFoundError:
        print("[ERROR] snake_qnet.pth not found!")
        return

    obs, _ = env.reset()
    clock = pygame.time.Clock()
    running = True

    while env.agents and running:
        # Handle Pygame window exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        actions = {}
        
        # 3. Neural Network Action Selection (Greedy / Pure Strategy)
        for agent in env.agents:
            state_t = torch.FloatTensor(obs[agent]).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = q_net(state_t)
            # Pick the action with the highest Q-value
            actions[agent] = q_vals.argmax(dim=1).item()

        # Step Environment & Render
        obs, rewards, terminations, truncations, _ = env.step(actions)
        env.render()

        # Cap frame rate so you can watch comfortably
        clock.tick(12)

    env.close()
    print("Match finished!")

if __name__ == "__main__":
    visualize()