import numpy as np
import pygame

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]


class StackelbergAgent():
    def __init__(self, gamma, initial_state_coord, transition_matrix_initial_state):
        self.gamma = gamma # discount factor
        self.agent_state_coord = initial_state_coord
        
        self.path = [self.agent_state_coord] # list of states visited by agent
        self.line_path = [] # list of lines to draw path

    def reward_function(self, episode):
        """
        Compute cumulative reward from the episode
        episode = [(state, action, reward, next_state), ...]
        """
        T = len(episode) # number of steps to reach terminal state from current state
        if T == 1:
            return episode[0][2] # reward of the final step
        else:
            return sum([self.gamma**t * self.reward_function(episode[1:]) for t in range(T)]) # recursive call

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