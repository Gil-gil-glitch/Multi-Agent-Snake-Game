import pygame
from config import *
from map import ArenaMap
from snake import Snake
from bot import TacticalBot

class GameEngine:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("Consolas", 18, bold=True)
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("2v2 Multi-Agent Snake Arena (Phases 3 & 4)")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.map = ArenaMap()
        self.team_scores = {TEAM_CYAN: 0, TEAM_MAGENTA: 0}

        # Calculate dynamic start positions relative to any GRID_SIZE
        row_top = max(1, GRID_SIZE // 4)
        row_bottom = min(GRID_SIZE - 2, (GRID_SIZE * 3) // 4)
        col_left = max(1, GRID_SIZE // 4)
        col_right = min(GRID_SIZE - 2, (GRID_SIZE * 3) // 4)

        # Spawn 4 AI Snakes with proportional grid coordinates
        self.p1 = Snake(1, TEAM_CYAN, start_pos=(row_top, col_left), start_dir=DIR_RIGHT, color=COLOR_P1, is_bot=True)
        self.p2 = Snake(2, TEAM_CYAN, start_pos=(row_bottom, col_left), start_dir=DIR_RIGHT, color=COLOR_P2, is_bot=True)

        self.p3 = Snake(3, TEAM_MAGENTA, start_pos=(row_top, col_right), start_dir=DIR_LEFT, color=COLOR_P3, is_bot=True)
        self.p4 = Snake(4, TEAM_MAGENTA, start_pos=(row_bottom, col_right), start_dir=DIR_LEFT, color=COLOR_P4, is_bot=True)

        self.snakes = [self.p1, self.p2, self.p3, self.p4]
        self.bots = {s.player_id: TacticalBot(s.player_id) for s in self.snakes if s.is_bot}

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and self.p1.alive:
                if event.key in P1_KEYS:
                    self.p1.set_direction(P1_KEYS[event.key])
        return True

    def update(self):
        # 1. Query AI Bots for moves
        for snake in self.snakes:
            if snake.alive and snake.is_bot:
                bot = self.bots[snake.player_id]
                action = bot.choose_action(snake, self)
                snake.set_direction(action)

        # Spawn food using MAX_FOOD from config (defaulting to 5 if undefined)
        food_target = globals().get('MAX_FOOD', 5)
        all_occupied = set(p for s in self.snakes if s.alive for p in s.body)
        self.map.spawn_food(all_occupied, target_count=food_target)

        # 2. Movement & Collision Engine
        for snake in self.snakes:
            if not snake.alive:
                continue

            # Water slowdown
            if self.map.get_tile(snake.body[0]) == TILE_WATER:
                snake.water_delay = not snake.water_delay
                if snake.water_delay:
                    continue

            next_head = snake.get_next_head()

            # Find if colliding with another snake
            target_snake = None
            for other in self.snakes:
                if other.alive and next_head in other.body:
                    target_snake = other
                    break

            if target_snake:
                if target_snake.team_id == snake.team_id:
                    # FRIENDLY FIRE OFF: Teammate safety bounce
                    continue
                else:
                    # ENEMY COLLISION -> Execute Push Mechanics
                    push_dir = snake.next_direction
                    pushed_head = (next_head[0] + push_dir[0], next_head[1] + push_dir[1])
                    pushed_tile = self.map.get_tile(pushed_head)

                    if pushed_tile in (TILE_LAVA, TILE_WALL):
                        # ENEMY SHOVED INTO HAZARD! Kill Enemy + Reward Team
                        target_snake.alive = False
                        self.team_scores[snake.team_id] += REWARD_KILL
                        self._resolve_step(snake, next_head)
                    else:
                        # Shove Enemy onto safe cell
                        target_snake.shift_entire_snake(push_dir)
                        self._resolve_step(snake, next_head)
            else:
                self._resolve_step(snake, next_head)

    def _resolve_step(self, snake, next_head):
        tile = self.map.get_tile(next_head)

        if tile in (TILE_LAVA, TILE_WALL):
            snake.alive = False
            return

        grow = False
        if next_head in self.map.foods:
            self.map.foods.remove(next_head)
            self.team_scores[snake.team_id] += REWARD_FOOD
            grow = True

        snake.move(next_head, grow=grow)

    def render(self):
        self.screen.fill(COLOR_GRASS)

        # Render Map Tiles (Shifted down by HUD_HEIGHT)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                tile = self.map.grid[r, c]
                if tile != TILE_GRASS:
                    color = COLOR_GRASS
                    if tile == TILE_FARM: color = COLOR_FARM
                    elif tile == TILE_WATER: color = COLOR_WATER
                    elif tile == TILE_LAVA: color = COLOR_LAVA
                    elif tile == TILE_WALL: color = COLOR_WALL

                    rect = (c * CELL_SIZE, r * CELL_SIZE + HUD_HEIGHT, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, color, rect)

        # Render Food
        for r, c in self.map.foods:
            rect = (c * CELL_SIZE, r * CELL_SIZE + HUD_HEIGHT, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, COLOR_FOOD, rect)

        # Render Snakes
        for snake in self.snakes:
            if not snake.alive:
                continue
            for i, (r, c) in enumerate(snake.body):
                rect = (c * CELL_SIZE, r * CELL_SIZE + HUD_HEIGHT, CELL_SIZE, CELL_SIZE)
                color = snake.color if i == 0 else (max(0, snake.color[0]-40), max(0, snake.color[1]-40), max(0, snake.color[2]-40))
                pygame.draw.rect(self.screen, color, rect)

        # Render Top HUD Banner
        hud_rect = (0, 0, WINDOW_SIZE[0], HUD_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_HUD, hud_rect)

        cyan_alive = sum(1 for s in self.snakes if s.team_id == TEAM_CYAN and s.alive)
        mag_alive = sum(1 for s in self.snakes if s.team_id == TEAM_MAGENTA and s.alive)

        txt_cyan = self.font.render(f"CYAN TEAM: {self.team_scores[TEAM_CYAN]} PTS ({cyan_alive} Alive)", True, COLOR_P1)
        txt_mag = self.font.render(f"MAGENTA TEAM: {self.team_scores[TEAM_MAGENTA]} PTS ({mag_alive} Alive)", True, COLOR_P3)

        self.screen.blit(txt_cyan, (15, 10))
        self.screen.blit(txt_mag, (WINDOW_SIZE[0] - txt_mag.get_width() - 15, 10))

        pygame.display.flip()
        self.clock.tick(FPS)