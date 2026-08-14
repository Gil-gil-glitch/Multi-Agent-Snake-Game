import pygame
from config import *
from map import ArenaMap
from snake import Snake

class GameEngine:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Multi-Agent Snake Arena - Phase 1 & 2 Sandbox")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.map = ArenaMap()
        # Spawn Player 1 (Cyan) near left farm entrance
        self.p1 = Snake(1, start_pos=(50, 30), start_dir=DIR_RIGHT, color=COLOR_SNAKE1)
        # Spawn Player 2 (Magenta) near right farm entrance
        self.p2 = Snake(2, start_pos=(50, 70), start_dir=DIR_LEFT, color=COLOR_SNAKE2)
        self.snakes = [self.p1, self.p2]

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in P1_KEYS and self.p1.alive:
                    self.p1.set_direction(P1_KEYS[event.key])
                if event.key in P2_KEYS and self.p2.alive:
                    self.p2.set_direction(P2_KEYS[event.key])
        return True

    def update(self):
        all_occupied = set(self.p1.body + self.p2.body)
        self.map.spawn_food(all_occupied)

        for snake in self.snakes:
            if not snake.alive:
                continue

            # --- Water Hazard Logic (Slow down movement) ---
            current_tile = self.map.get_tile(snake.body[0])
            if current_tile == TILE_WATER:
                snake.water_delay = not snake.water_delay
                if snake.water_delay:
                    continue  # Skip move tick on water

            next_head = snake.get_next_head()
            other_snake = self.p2 if snake == self.p1 else self.p1

            # --- Push / Bump Collision Engine ---
            if other_snake.alive and next_head in other_snake.body:
                push_dir = snake.next_direction
                # Attempt to shove other snake 1 cell in push direction
                if self._try_push_snake(other_snake, push_dir):
                    # Successfully pushed! Move aggressive snake into freed cell
                    self._resolve_snake_step(snake, next_head)
                else:
                    # Target couldn't be shoved (e.g. backed against wall) -> Aggressor dies
                    snake.alive = False
            else:
                self._resolve_snake_step(snake, next_head)

    def _try_push_snake(self, defender, push_dir):
        """Attempts to push the defender snake. Kills defender if pushed into Lava/Wall."""
        shoved_positions = [(r + push_dir[0], c + push_dir[1]) for r, c in defender.body]
        
        # Check if the pushed head or body hits deadly terrain
        for pos in shoved_positions:
            tile = self.map.get_tile(pos)
            if tile in (TILE_LAVA, TILE_WALL):
                defender.alive = False  # Defender died from being shoved into hazard!
                return True

        defender.shift_entire_snake(push_dir)
        return True

    def _resolve_snake_step(self, snake, next_head):
        tile = self.map.get_tile(next_head)

        # Lava or Wall Death
        if tile in (TILE_LAVA, TILE_WALL):
            snake.alive = False
            return

        # Food Eating
        grow = False
        if next_head in self.map.foods:
            self.map.foods.remove(next_head)
            snake.score += 10
            grow = True

        snake.move(next_head, grow=grow)

    def render(self):
        self.screen.fill(COLOR_GRASS)

        # Render Map Tiles
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                tile = self.map.grid[r, c]
                if tile != TILE_GRASS:
                    color = COLOR_GRASS
                    if tile == TILE_FARM: color = COLOR_FARM
                    elif tile == TILE_WATER: color = COLOR_WATER
                    elif tile == TILE_LAVA: color = COLOR_LAVA
                    elif tile == TILE_WALL: color = COLOR_WALL
                    
                    rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, color, rect)

        # Render Food
        for r, c in self.map.foods:
            rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, COLOR_FOOD, rect)

        # Render Snakes
        for snake in self.snakes:
            if not snake.alive:
                continue
            for i, (r, c) in enumerate(snake.body):
                rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                # Head is drawn slightly brighter
                color = snake.color if i == 0 else (max(0, snake.color[0]-40), max(0, snake.color[1]-40), max(0, snake.color[2]-40))
                pygame.draw.rect(self.screen, color, rect)

        pygame.display.flip()
        self.clock.tick(FPS)