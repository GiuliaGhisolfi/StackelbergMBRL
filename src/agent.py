import numpy as np
import pygame

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]
ACTION_MAP = { #TODO: togliere o cambiare mappatura
    0: 'up', 
    1: 'down', 
    2: 'left', 
    3: 'right'
}

class Agent():
    def __init__(self, initial_state_coord, transition_matrix_initial_state,
            gamma, actions_list_initial_state):
        self.model = ModelAgent(initial_state_coord, transition_matrix_initial_state)
        self.policy = PolicyAgent(gamma, initial_state_coord, actions_list_initial_state)

        self.initial_state = initial_state_coord
        self.agent_state = initial_state_coord

        self.path = [self.agent_state] # list of states visited by agent
        self.line_path = [] # list of lines to draw path
    
    def render(self, window, block_pixel_size):
        # draw agent
        pygame.draw.rect(window, 
            (0, 0, 255), 
            pygame.Rect(
                block_pixel_size * self.agent_state[0] + int(block_pixel_size/5),
                block_pixel_size * self.agent_state[1] + int(block_pixel_size/5),
                block_pixel_size - 2 * int(block_pixel_size/5),
                block_pixel_size - 2 * int(block_pixel_size/5),
            ),
        )

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
        
        # display update
        pygame.display.update()

class ModelAgent():
    def __init__(self, initial_state_coord, transition_matrix_initial_state):
        self.model_action_space = [0, 1, 2, 3] # [up, down, left, right]
        self.model_state_space = dict() # discrete: {(x,y): [possible actions]}

        self.update_model_space(initial_state_coord, transition_matrix_initial_state)

    def transition_function(self, state, action):
        #return next_state
        pass

    def update_model(self, state, transition_matrix):
        # update state space and action
        self.update_model_space(state, transition_matrix)

        # update transition matrix: SAME THAT UPDATE POLICY
        #TODO
    
    def update_model_space(self, state, transition_matrix):
        #transition_matrix = env.p[:, state, :]
        # state as a tuple (x,y)
        if state not in self.model_state_space.keys():
            self.model_state_space[state] = np.where(transition_matrix != 0)[1] # possible actions from state


class PolicyAgent():
    def __init__(self, gamma, initial_state_coord, actions_list_initial_state):
        self.gamma = gamma # discount factor
        self.initial_state = initial_state_coord
        self.initialize_policy(initial_state_coord, actions_list_initial_state)
    
    def initialize_policy(self, initial_state, actions_list_initial_state):
        self.policy = dict() # {(x,y): [action's probability]}

        actions_probability_list = np.zeros(N_ACTIONS)
        actions_probability_list[actions_list_initial_state] = 1/len(actions_list_initial_state)

        self.policy[initial_state] = actions_probability_list
    
    def take_action(self, state):
        return np.random.choice(ACTIONS_LIST, self.policy[state])

    def update_policy(self, state):
        # update self.policy
        pass
    
    def reward_function(self, state, action, next_state):
        #return cumulative_reward
        pass