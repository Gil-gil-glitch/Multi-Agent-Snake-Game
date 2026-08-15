import pygame

# --- Grid & Window Config (Smaller Map = More Combat) ---
GRID_SIZE = 40          # Reduced from 100 to force close-quarters fights
CELL_SIZE = 20          # Scaled up so window size stays 800x800
HUD_HEIGHT = 40         # Extra pixels at top for score banner
WINDOW_SIZE = (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + HUD_HEIGHT)
FPS = 15                # Game rendering speed

# --- Gameplay & Combat Balancers ---
MAX_FOOD = 3            # Keeps food scarce to trigger battles
MAX_HUNGER = 50         # Tighter starvation timer forces movement
INITIAL_SNAKE_LEN = 5   # Starting longer makes body-blocking possible immediately

# --- Terrain Generation (Biome patch counts, scaled off GRID_SIZE) ---
LAVA_PATCHES = max(1, GRID_SIZE // 15)
LAVA_RADIUS_RANGE = (2, 4)
WATER_PATCHES = max(1, GRID_SIZE // 10)
WATER_RADIUS_RANGE = (2, 5)
WALL_PATCHES = max(1, GRID_SIZE // 12)
WALL_RADIUS_RANGE = (1, 2)
FARM_PATCHES = max(1, GRID_SIZE // 10)
FARM_RADIUS_RANGE = (2, 4)
SPAWN_CLEAR_RADIUS = 2   # Keeps hazards from spawning on top of a snake's starting square

# --- Match Setup ---
# Assign one of four "slots" (spawn corners / colors) to a player_id, a team,
# and an algorithm. Snakes on the same "team" don't collide with each other
# and share a score - that's how you build a 2v2. Give every snake a unique
# team id (e.g. 1, 2, 3, 4) to get a free-for-all instead, or mix and match
# to have some snakes team up while others go solo.
#
# algorithm options: "human" (arrow keys, player 1 only), "tactical"
# (A* food-seeking + push-kill bot), "anmitsu" (flood-fill/hunger-phase bot),
# "random" (safe random mover, good as a weak baseline)
SNAKE_SETUP = {
    1: {"team": 1, "algorithm": "human"},
    2: {"team": 1, "algorithm": "tactical"},
    3: {"team": 2, "algorithm": "anmitsu"},
    4: {"team": 2, "algorithm": "tactical"},
}

# --- Score Values (High Kill Bounty) ---
REWARD_FOOD = 5         # Reduced passive eating reward
REWARD_KILL = 150       # Massive reward bonus for eliminating enemies

# --- Tile Types ---
TILE_GRASS = 0
TILE_FARM = 1
TILE_WATER = 2
TILE_LAVA = 3
TILE_WALL = 4

# --- Colors ---
COLOR_GRASS = (30, 40, 30)
COLOR_FARM = (60, 100, 50)
COLOR_WATER = (30, 80, 160)
COLOR_LAVA = (220, 50, 20)
COLOR_WALL = (70, 70, 80)
COLOR_FOOD = (255, 215, 0)
COLOR_HUD = (20, 20, 25)

# --- Teams & Snake Colors ---
TEAM_CYAN = 1
TEAM_MAGENTA = 2

COLOR_P1 = (0, 255, 220)    # Cyan Main
COLOR_P2 = (0, 160, 255)    # Cyan Ally (Blue)
COLOR_P3 = (255, 0, 128)    # Magenta Main
COLOR_P4 = (255, 100, 0)    # Magenta Ally (Orange)

# --- Directions & Controls ---
DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)

ALL_DIRECTIONS = [DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT]

P1_KEYS = {
    pygame.K_UP: DIR_UP,
    pygame.K_DOWN: DIR_DOWN,
    pygame.K_LEFT: DIR_LEFT,
    pygame.K_RIGHT: DIR_RIGHT
}