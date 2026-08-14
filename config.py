import pygame

# Grid & Window Config
GRID_SIZE = 100
CELL_SIZE = 8
HUD_HEIGHT = 40  # Extra pixels at top for score banner
WINDOW_SIZE = (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + HUD_HEIGHT)
FPS = 12

# Tile Types
TILE_GRASS = 0
TILE_FARM = 1
TILE_WATER = 2
TILE_LAVA = 3
TILE_WALL = 4

# Colors
COLOR_GRASS = (30, 40, 30)
COLOR_FARM = (60, 100, 50)
COLOR_WATER = (30, 80, 160)
COLOR_LAVA = (220, 50, 20)
COLOR_WALL = (70, 70, 80)
COLOR_FOOD = (255, 215, 0)
COLOR_HUD = (20, 20, 25)

# Teams
TEAM_CYAN = 1
TEAM_MAGENTA = 2

# 4 Snake Colors (Primary & Secondary for each team)
COLOR_P1 = (0, 255, 220)    # Cyan Main
COLOR_P2 = (0, 160, 255)    # Cyan Ally (Blue)
COLOR_P3 = (255, 0, 128)    # Magenta Main
COLOR_P4 = (255, 100, 0)    # Magenta Ally (Orange)

# Score Values
REWARD_FOOD = 10
REWARD_KILL = 50

# Directions
DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)

ALL_DIRECTIONS = [DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT]

# Keybindings (Player 1 Controls Cyan Leader)
P1_KEYS = {
    pygame.K_UP: DIR_UP,
    pygame.K_DOWN: DIR_DOWN,
    pygame.K_LEFT: DIR_LEFT,
    pygame.K_RIGHT: DIR_RIGHT
}