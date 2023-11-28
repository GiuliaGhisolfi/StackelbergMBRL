import pygame


class Agent():
    def __init__(self, initial_state_coord):
        self.agent_state_coord = initial_state_coord
        
        self.path = [self.agent_state_coord] # list of states visited by agent
        self.line_path = [] # list of lines to draw path

    def reset(self):
        self.agent_state_coord = self.path[0]
        self.path = [self.agent_state_coord]
        self.line_path = []

    def render(self, window, block_pixel_size):
        # draw path
        if block_pixel_size % 2:
            line_width = 2
        else: 
            line_width = 3

        self.line_path.append(
            (self.path[-1][0] * block_pixel_size + int(block_pixel_size/2),
            self.path[-1][1] * block_pixel_size + int(block_pixel_size/2))
        )
        if len(self.line_path) > 1:
            pygame.draw.lines(
                window,
                (0, 0, 0),
                False,
                self.line_path,
                line_width
            )

        # draw agent
        pygame.draw.rect(window, 
            (0, 0, 255), 
            pygame.Rect(
                block_pixel_size * self.agent_state_coord[0] + int(block_pixel_size/5),
                block_pixel_size * self.agent_state_coord[1] + int(block_pixel_size/5),
                block_pixel_size - 2 * int(block_pixel_size/5),
                block_pixel_size - 2 * int(block_pixel_size/5),
            ),
        )