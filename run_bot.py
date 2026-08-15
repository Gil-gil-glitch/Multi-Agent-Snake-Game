import pygame
from pettingzoo_env import SnakeArenaParallelEnv
from anmitsusnake_bot import AnmitsuSnakeBot

def run_match():
    env = SnakeArenaParallelEnv(render_mode="human")
    obs, _ = env.reset()
    
    # Assign heuristic bots to all active snakes
    bots = {agent: AnmitsuSnakeBot(agent_id=agent) for agent in env.possible_agents}
    
    clock = pygame.time.Clock()
    running = True

    while env.agents and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        actions = {agent: bots[agent].get_action(obs[agent]) for agent in env.agents}
        obs, rewards, terminations, truncations, _ = env.step(actions)
        
        env.render()
        clock.tick(15)

    env.close()

if __name__ == "__main__":
    run_match()