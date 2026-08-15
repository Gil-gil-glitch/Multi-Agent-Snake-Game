import pygame
from config import *
from map import ArenaMap
from snake import Snake
from algorithms import BOT_REGISTRY

class GameEngine:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("Consolas", 18, bold=True)
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Multi-Agent Snake Arena")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.map = ArenaMap()

        # Calculate dynamic start positions relative to any GRID_SIZE
        row_top = max(1, GRID_SIZE // 4)
        row_bottom = min(GRID_SIZE - 2, (GRID_SIZE * 3) // 4)
        col_left = max(1, GRID_SIZE // 4)
        col_right = min(GRID_SIZE - 2, (GRID_SIZE * 3) // 4)

        # Four fixed spawn corners (position, facing direction), matched by
        # index to sorted player_id order in SNAKE_SETUP.
        spawn_layout = [
            (row_top, col_left, DIR_RIGHT),
            (row_bottom, col_left, DIR_RIGHT),
            (row_top, col_right, DIR_LEFT),
            (row_bottom, col_right, DIR_LEFT),
        ]
        colors = [COLOR_P1, COLOR_P2, COLOR_P3, COLOR_P4]

        self.snakes = []
        self.bots = {}
        self.human_player = None

        for idx, player_id in enumerate(sorted(SNAKE_SETUP.keys())):
            setup = SNAKE_SETUP[player_id]
            team_id = setup["team"]
            algorithm = setup["algorithm"]
            r, c, start_dir = spawn_layout[idx % len(spawn_layout)]
            is_bot = algorithm != "human"

            snake = Snake(player_id, team_id, start_pos=(r, c), start_dir=start_dir,
                          color=colors[idx % len(colors)], is_bot=is_bot)
            self.snakes.append(snake)

            # Guarantee no one spawns on top of lethal terrain
            self.map.clear_area((r, c))

            if is_bot:
                bot_cls = BOT_REGISTRY[algorithm]
                self.bots[player_id] = bot_cls(player_id)
            else:
                self.human_player = snake

        # Score is tracked per team id - give every snake its own team id for
        # free-for-all, or share a team id to make a squad (2v2, 3v1, etc).
        self.team_ids = sorted(set(s.team_id for s in self.snakes))
        self.team_scores = {tid: 0 for tid in self.team_ids}

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and self.human_player and self.human_player.alive:
                if event.key in P1_KEYS:
                    self.human_player.set_direction(P1_KEYS[event.key])
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

        # Render Top HUD Banner (one label per team, spaced left to right)
        hud_rect = (0, 0, WINDOW_SIZE[0], HUD_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_HUD, hud_rect)

        x_offset = 15
        for tid in self.team_ids:
            team_snakes = [s for s in self.snakes if s.team_id == tid]
            alive_count = sum(1 for s in team_snakes if s.alive)
            team_color = team_snakes[0].color
            label = f"TEAM {tid}: {self.team_scores[tid]} PTS ({alive_count}/{len(team_snakes)} Alive)"
            txt = self.font.render(label, True, team_color)
            self.screen.blit(txt, (x_offset, 10))
            x_offset += txt.get_width() + 30

        pygame.display.flip()
        self.clock.tick(FPS)