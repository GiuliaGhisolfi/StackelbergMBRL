import numpy as np

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]
WALLS_MAP = {
    0: np.array([3, 1, 2, 0]), # down:  left, up, right, down
    1: np.array([2, 0, 3, 1]), # up:    right, down, left, up
    2: np.array([0, 3, 1, 2]), # right: down, left, up, right
    3: np.array([1, 2, 0, 3])  # left:  up, right, down, left
}

class PolicyAgent():
    def __init__(self, transition_matrix_initial_state):
        self.policy = dict() # {walls[-,-,-]: probability distribution over actions}

        action = self.__compute_fitizial_first_action(transition_matrix_initial_state)
        self.update_policy(action=action, transition_matrix=transition_matrix_initial_state)
    
    def update_policy(self, action, transition_matrix):
        not_walls = self.__compute_walls_from_transition_matrix(action, transition_matrix)
        if not_walls not in self.policy.keys():
            self.policy[not_walls] = not_walls / np.sum(not_walls)
        else:
            # update policy using idk what but consider reward
            pass
    
    def compute_next_action(self, action, transition_matrix):
        not_walls = self.__update_state_space(action, transition_matrix)
        return np.random.choice(ACTIONS_LIST, p=self.policy[not_walls])
    
    def __update_state_space(self, action, transition_matrix):
        not_walls = self.__compute_walls_from_transition_matrix(action, transition_matrix)
        if not_walls not in self.policy.keys():
            self.policy[not_walls] = not_walls / np.sum(not_walls)
        return not_walls
    
    def __compute_walls_from_transition_matrix(self, action, transition_matrix):
        # compute walls from the agent's point of view: 0 if there is a wall, 1 otherwise
        # wall 0: agent's left
        # wall 1: agent's below
        # wall 2: agent's right
        # wall 3: agent's above
        env_not_walls = np.sum(transition_matrix, axis=0)
        return env_not_walls[WALLS_MAP[action]]
    
    def __compute_fitizial_first_action(self, transition_matrix_initial_state):
        # compute a fitizial action to possibly arrive in initial state
        possible_actions = np.sum(transition_matrix_initial_state, axis=0)
        return np.random.choice(np.where(possible_actions != 0)[0])