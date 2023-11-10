import numpy as np

def initalize_agents(gamma, alpha, beta, k):
    pass

class Agent:
    def __init__(self, gamma, alpha, beta, k):
        self.gamma = gamma # discount factor
        self.alpha = alpha # learning rate for policy improvment
        self.beta = beta # learning rate for model update
        self.k = k # number of episodes to run for each epoch

        # initialize data buffer
        self.data_buffer = [] # list of episodes, each episode is a list of tuples (state, action, reward, next_state)

class ModelAgent(Agent):

    def __init__(self):
        super().__init__()
        self.action_space = None # to update during training
        self.state_space = None # discrete, from maze class        
        pass

    def reward_function(self, state, action, next_state):
        #return reward
        pass

    def transition_function(self, state, action):
        #return next_state
        pass

    def update_model(self):
        # update transition matrix
        # update reward function
        # update state space
        pass

class PolicyAgent(Agent):

    def __init__(self):
        super().__init__()