import numpy as np

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]


class ModelAgent():
    def __init__(self, gamma, transition_matrix_initial_state):
        self.gamma = gamma
        self.transition_matrix = transition_matrix_initial_state