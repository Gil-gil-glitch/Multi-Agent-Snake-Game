from game import GameEngine

if __name__ == "__main__":
    game = GameEngine()
    running = True
    print("--- 100x100 Multi-Agent Arena (Phase 1 & 2) ---")
    print("Player 1 (Cyan): Arrow Keys")
    print("Player 2 (Magenta): WASD Keys")
    
    while running:
        running = game.handle_input()
        game.update()
        game.render()

    game.reset()