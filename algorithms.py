import random

from config import ALL_DIRECTIONS, TILE_LAVA, TILE_WALL
from bot import TacticalBot
from anmitsusnake_bot import AnmitsuSnakeBot
from observation import build_observation


class RandomBot:
    """Weak baseline: picks a random move that isn't instant suicide.
    Useful as a control to benchmark the smarter algorithms against."""

    def __init__(self, player_id):
        self.player_id = player_id

    def choose_action(self, bot_snake, game_state):
        head = bot_snake.body[0]
        opposite = (-bot_snake.direction[0], -bot_snake.direction[1])
        candidates = [d for d in ALL_DIRECTIONS if d != opposite]
        random.shuffle(candidates)

        for move in candidates:
            nr, nc = head[0] + move[0], head[1] + move[1]
            tile = game_state.map.get_tile((nr, nc))
            blocked = any((nr, nc) in s.body for s in game_state.snakes if s.alive)
            if tile not in (TILE_LAVA, TILE_WALL) and not blocked:
                return move

        # No safe move found - just hold direction and accept the risk
        return bot_snake.direction


class AnmitsuBotAdapter:
    """AnmitsuSnakeBot was built around a PettingZoo-style obs tensor and
    returns an action index. This adapter builds that tensor from the live
    game state each tick and translates the index back into a direction
    vector, so it can be dropped into the same bot slot as TacticalBot."""

    def __init__(self, player_id):
        self.player_id = player_id
        self._bot = AnmitsuSnakeBot(agent_id=f"snake_{player_id}")

    def choose_action(self, bot_snake, game_state):
        obs = build_observation(game_state.map, game_state.snakes, bot_snake)
        action_idx = self._bot.get_action(obs)
        return ALL_DIRECTIONS[action_idx]


# Name -> constructor(player_id). "human" is handled separately in game.py
# since it reads keyboard input rather than choosing an action itself.
BOT_REGISTRY = {
    "tactical": TacticalBot,
    "anmitsu": AnmitsuBotAdapter,
    "random": RandomBot,
}