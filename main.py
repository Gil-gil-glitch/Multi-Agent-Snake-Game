from game import GameEngine

if __name__ == "__main__":
    game = GameEngine()
    running = True


    print("      2v2 STRATEGIC SNAKE ARENA - BATTLE START    ")
    print("You control Player 1 (Bright Cyan) with Arrow Keys.")
    print("Your Teammate (Blue) is an AI Bot.")
    print("Opponents: Magenta & Orange AI Bots.")
    print("Tactics: Push enemy snakes into Lava for +50 PTS!")

    while running:
        running = game.handle_input()
        game.update()
        game.render()

    pygame.quit()