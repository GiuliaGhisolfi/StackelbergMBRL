import numpy as np

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]

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