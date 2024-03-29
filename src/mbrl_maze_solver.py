import pygame
from src.environment.environment import Environment
from src.agents.baseline import Baseline
from src.agents.stackelberg_agent import StackelbergAgent

class MBRLMazeSolver():

    def __init__(self, maze_width, maze_height, max_epochs, algorithm='baseline',
        n_episodes_per_iteration=100, gamma=0.9, alpha=0.01, beta=0.01, render=True, render_wait=0,
        policy_path='src/saved_policy/PAL_policy.json', 
        states_space_path='src/saved_policy/PAL_states_space.json'):
        # TODO: parameters check
        self.algorithm = algorithm # 'PAL' or 'MAL' or 'baseline'
        self.max_epochs = max_epochs

        self.render_bool = render
        self.render_wait = render_wait

        # initialize environment
        self.env = Environment(
            maze_width=maze_width,
            maze_height=maze_height
            )
        
        # initialize agent: policy and model
        if self.algorithm == 'baseline':
            self.agent = Baseline(
                initial_state_coord=self.env.initial_state_coord,
                transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],
                )
        else:
            self.agent = StackelbergAgent(
                initial_state_coord=self.env.initial_state_coord,
                policy_path=policy_path,
                states_space_path=states_space_path,
                transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],
                )
        
        if self.render_bool:
            self.render(self.render_wait)

    def run(self):
        # run algorithm
        if self.algorithm == 'baseline':
            self.run_baseline()
        else:
            self.run_stackelberg()

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
            if self.render_bool:
                self.render(self.render_wait)
            
            if self.agent.agent_state_coord == self.env.terminal_state_coord:
                print('Agent reached terminal state in {} steps'.format(epoch))
                break # stop if agent reached terminal state
                
    
    def run_stackelberg(self):
        # run algorithm
        action = self.agent.fittizial_first_action
        for epoch in range(self.max_epochs):
            # execute policy
            action = self.agent.take_action(
                action=action,
                transition_matrix=self.env.p[:, self.env.state, :]
                )
            
            # take a step in the environment
            self.env.step(action)

            # update current state
            self.agent.update_agent_parameters(
                next_state_coord=self.env.coordinates_from_state(self.env.state)
            )
            
            # render environment and agent
            if self.agent.agent_state_coord == self.env.terminal_state_coord:
                self.render(2000)
            elif self.render_bool:
                self.render(self.render_wait)
            
            
            if self.agent.agent_state_coord == self.env.terminal_state_coord:
                print('Agent reached terminal state in {} steps'.format(epoch))
                break

    def render(self, wait=0):
        # Render environment and agent using pygame
        self.env.render()
        self.agent.render(self.env.window, self.env.block_pixel_size)

        # display update
        pygame.display.update()
        pygame.time.wait(wait) # wait (ms)