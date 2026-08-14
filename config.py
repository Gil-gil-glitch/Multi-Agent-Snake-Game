import pygame

# Grid & Window Config
GRID_SIZE = 100
CELL_SIZE = 8
WINDOW_SIZE = (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
FPS = 12  # Game tick speed

# Tile Types
TILE_GRASS = 0
TILE_FARM = 1
TILE_WATER = 2
TILE_LAVA = 3
TILE_WALL = 4

# Colors (RGB)
COLOR_GRASS = (30, 40, 30)
COLOR_FARM = (60, 100, 50)
COLOR_WATER = (30, 80, 160)
COLOR_LAVA = (220, 50, 20)
COLOR_WALL = (70, 70, 80)
COLOR_FOOD = (255, 215, 0)

COLOR_SNAKE1 = (0, 255, 200)   # Cyan
COLOR_SNAKE2 = (255, 0, 128)   # Magenta

# Directions (Row, Col)
DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)

# Keybindings
P1_KEYS = {
    pygame.K_UP: DIR_UP,
    pygame.K_DOWN: DIR_DOWN,
    pygame.K_LEFT: DIR_LEFT,
    pygame.K_RIGHT: DIR_RIGHT
}

P2_KEYS = {
    pygame.K_w: DIR_UP,
    pygame.K_s: DIR_DOWN,
    pygame.K_a: DIR_LEFT,
    pygame.K_d: DIR_RIGHT
}