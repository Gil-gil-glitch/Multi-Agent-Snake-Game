from config import *

class Snake:
    def __init__(self, player_id, team_id, start_pos, start_dir, color, is_bot=False):
        self.player_id = player_id
        self.team_id = team_id
        self.body = [start_pos, (start_pos[0] - start_dir[0], start_pos[1] - start_dir[1])]
        self.direction = start_dir
        self.next_direction = start_dir
        self.color = color
        self.is_bot = is_bot
        self.alive = True
        self.water_delay = False

    def set_direction(self, new_dir):
        # Prevent 180-degree self-reversal
        if (new_dir[0] + self.direction[0] != 0) or (new_dir[1] + self.direction[1] != 0):
            self.next_direction = new_dir

    def get_next_head(self, custom_dir=None):
        d = custom_dir if custom_dir else self.next_direction
        return (self.body[0][0] + d[0], self.body[0][1] + d[1])

    def move(self, new_head, grow=False):
        self.direction = self.next_direction
        self.body.insert(0, new_head)
        if not grow:
            self.body.pop()

    def shift_entire_snake(self, shift_vector):
        """
        Shoves the snake body by a vector (Push mechanic).
        """
        self.body = [(r + shift_vector[0], c + shift_vector[1]) for r, c in self.body]