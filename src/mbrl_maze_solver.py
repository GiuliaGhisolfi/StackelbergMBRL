import numpy as np
import pygame
from src.environment.environment import Environment
from src.algorithms.baseline import Baseline
from src.algorithms.MAL import MAL
from src.algorithms.PAL import PAL

class MBRLMazeSolver():

    def __init__(self, maze_width, maze_height, max_epochs, algorithm='baseline',
        n_episodes_per_iteration=100, gamma=0.9, alpha=0.01, beta=0.01):
        self.algorithm = algorithm # 'PAL' or 'MAL' or 'baseline'
        self.max_epochs = max_epochs

        # initialize environment
        self.env = Environment(
            maze_width=maze_width,
            maze_height=maze_height
            )
        
        # initialize agent: policy and model
        if self.algorithm == 'baseline':
            self.agent = Baseline(
                gamma=gamma,
                initial_state_coord=self.env.initial_state_coord,
                transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],
                )
        elif self.algorithm == 'MAL':
            self.agent = MAL(
                gamma=gamma,
                initial_state_coord=self.env.initial_state_coord,
                transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],
                learning_rate=beta,
                n_episodes_per_iteration=n_episodes_per_iteration
                )
        elif self.algorithm == 'PAL':
            self.agent = PAL(
                gamma=gamma,
                initial_state_coord=self.env.initial_state_coord,
                transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],
                learning_rate=alpha,
                n_episodes_per_iteration=n_episodes_per_iteration
                )
        
        self.render()

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
            next_action = self.agent.take_action()
            
            # take a step in the environment
            self.env.step(next_action)

            # update current state
            self.previous_state = self.agent.agent_state_coord

            # update agent
            self.agent.update_agent_parameters(
                    next_state_coord=self.env.coordinates_from_state(self.env.state),
                    previous_state_coord=self.previous_state,
                    previous_state_cardinality=self.env.state_from_coordinates(self.previous_state[0], self.previous_state[1]), 
                    transition_matrix=self.env.p[:, self.env.state, :],
                    reached_terminal_state=(self.env.coordinates_from_state(self.env.state) == self.env.terminal_state_coord)
                )
            
            # render environment and agent
            self.render()
            
            if self.agent.agent_state_coord == self.env.terminal_state_coord:
                print('Agent reached terminal state in {} steps'.format(epoch))
                break # stop if agent reached terminal state
                
    
    def run_PAL(self):
        pass

    def run_MAL(self):
        pass

    def render(self, wait=0):
        # Render environment and agent using pygame
        self.env.render()
        self.agent.render(self.env.window, self.env.block_pixel_size)

        # display update
        pygame.display.update()
        pygame.time.wait(wait) # wait (ms)