import numpy as np

class ModelAgent():

    def __init__(self):
        self.model_action_space = [0, 1, 2, 3] # [up, down, left, right]
        self.model_state_space = dict() # discrete: {(x,y): [possible actions]}
        pass

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

    def __init__(self, gamma):
        self.gamma = gamma # discount factor
    
    def initialize_policy(self, initial_state):
        #return policy
        pass
    
    def take_action(self, state):
        #return action
        pass

    def update_policy(self, state):
        # update policy
        pass
    
    def reward_function(self, state, action, next_state):
        #return cumulative_reward
        pass