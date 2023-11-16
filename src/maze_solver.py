import numpy as np
import pygame

from src.environment import Environment
from src.agent import Agent
from src.algorithms.MAL import MAL
from src.algorithms.PAL import PAL

class MazeSolver():

    def __init__(self, maze_width, maze_height, gamma, algorithm='baseline', max_epochs=100, n_episodes_per_iteration=100):
        self.algorithm = algorithm # 'PAL' or 'MAL' or 'baseline'
        self.max_epochs = max_epochs
        self.n_episodes_per_iteration = n_episodes_per_iteration

        # initialize environment
        self.env = Environment(
            maze_width=maze_width,
            maze_height=maze_height
            )
        
        # initialize agent: policy and model
        self.agent = Agent(
            initial_state_coord=self.env.initial_state_coord, 
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],
            gamma=gamma, 
            actions_list_initial_state = np.where(self.env.p[:, self.env.initial_state, :] != 0)[1]
            )
        
        self.render()
        
    def render(self, wait=0):
        # Render environment and agent using pygame
        self.env.render()
        self.agent.render(self.env.window, self.env.block_pixel_size)

        # display update
        pygame.display.update()
        pygame.time.wait(wait) # wait (ms)

    def run(self):
        # run algorithm
        if self.algorithm == 'PAL':
            self.run_PAL()
        elif self.algorithm == 'MAL':
            self.run_MAL()
        elif self.algorithm == 'baseline':
            self.run_baseline()

    def run_baseline(self):
        # run algorithm
        for epoch in range(self.max_epochs):
            # execute policy
            next_action = self.agent.policy_agent.take_action(self.agent.agent_state_coord)
            
            # take a step in the environment
            self.env.step(next_action)

            # update current state
            previous_state = self.agent.agent_state_coord
            self.agent.agent_state_coord = self.env.coordinates_from_state(self.env.state)

            # update agent
            self.agent.update(self.agent.agent_state_coord, previous_state,
                previous_state_cardinality=self.env.state_from_coordinates(previous_state[0], previous_state[1]), 
                transition_matrix=self.env.p[:, self.env.state, :], 
                terminal_state_check=(self.agent.agent_state_coord == self.env.terminal_state_coord))
            
            # render environment and agent
            self.render()
            
            if self.agent.agent_state_coord == self.env.terminal_state_coord:
                break # stop if agent reached terminal state
    
    def run_PAL(self):
        pass

    def run_MAL(self):
        pass
