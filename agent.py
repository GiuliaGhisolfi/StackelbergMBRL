import numpy as np

class Agent:
    def __init__(self, env):
        self.current_state = None
        self.current_action = None
        pass

class Model(Agent):

    def __init__(self, env):
        super().__init__(env)
        self.action_space = None # to update during training
        self.state_space = None # discrete, from maze class
        self.transition_matrix = None
        self.reward = None
        self.gamma = None # discount factor
        pass

    def reward_function(self, state, action):
        pass

    def transition_function(self, state, action):
        pass

class Policy(Agent):

    def __init__(self, env):
        super().__init__(env)