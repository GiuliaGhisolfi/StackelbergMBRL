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
        self.policy = dict() # {state number: probability distribution over actions}
        self.states_space = dict() # {state number: walls[-,-,-]}

        self.__compute_fitizial_first_action(transition_matrix_initial_state)
        self.update_policy(action=self.fittizial_first_action, 
            transition_matrix=transition_matrix_initial_state)
    
    def update_policy(self, action, transition_matrix):
        not_walls = self.__compute_walls_from_transition_matrix(action, transition_matrix)
        
        if [(not_walls == w).all() for w in self.states_space.values()]:
            # TODO: update policy using idk what but consider rewards
            pass
        else:
            self.states_space[len(self.states_space)] = not_walls
            self.policy[len(self.policy)] = not_walls / np.sum(not_walls)
    
    def compute_next_action(self, action, transition_matrix):
        state_number = self.__update_states_space(action, transition_matrix)
        action_agent_pov = np.random.choice(ACTIONS_LIST, p=self.policy[state_number])
        
        # map action from agent's point of view to environment's point of view
        return np.where(WALLS_MAP[action] == action_agent_pov)[0][0] #FIXME
    
    def reset_at_initial_state(self, transition_matrix_initial_state):
        action = self.__compute_fitizial_first_action(transition_matrix_initial_state)
        self.update_policy(action=action, transition_matrix=transition_matrix_initial_state)
    
    def __update_states_space(self, action, transition_matrix):
        not_walls = self.__compute_walls_from_transition_matrix(action, transition_matrix)
        if not [(not_walls == w).all() for w in self.states_space.values()]:
            self.states_space[len(self.states_space)] = not_walls
            self.policy[len(self.states_space)] = not_walls / np.sum(not_walls)
        for state_number, walls in self.states_space.items():
            if (not_walls == walls).all():
                return state_number
    
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
        self.fittizial_first_action = np.random.choice(np.where(possible_actions != 0)[0])