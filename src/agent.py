import numpy as np

class Agent:
    def __init__(self):
        self.current_state = None
        self.current_action = None
        pass

class ModelAgent(Agent):

    def __init__(self, gamma):
        super().__init__()
        self.action_space = None # to update during training
        self.state_space = None # discrete, from maze class
        self.transition_matrix = None
        self.reward = None
        self.gamma = gamma # discount factor
        pass

    def reward_function(self, state, action):
        pass

    def transition_function(self, state, action):
        pass

class PolicyAgent(Agent):

    def __init__(self):
        super().__init__()