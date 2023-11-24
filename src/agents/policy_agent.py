import numpy as np

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]
WALLS_MAP = {
    0: np.array([2, 0, 3, 1]), # up action:    right, down, left, up
    1: np.array([3, 1, 2, 0]), # down action:  left, up, right, down
    2: np.array([1, 2, 0, 3]), # left action:  down, left, up, right
    3: np.array([0, 3, 1, 2])  # right action: up, right, down, left
}

class PolicyAgent():
    def __init__(self, transition_matrix_initial_state: np.ndarray):
        self.policy = dict() # {state number: probability distribution over actions}
        self.states_space = dict() # {state number: walls[-,-,-]}

        self.__compute_fitizial_first_action(transition_matrix_initial_state)
        self.update_policy(action=self.fittizial_first_action, 
            transition_matrix=transition_matrix_initial_state)
    
    def update_policy(self, action:int, transition_matrix:np.ndarray):
        """
        Update policy with new state if it is not already present in the states space
        else update policy with new transition matrix and add new state to states space

        Args:
            action (int): action taken by the agent
            transition_matrix (np.ndarray): transition matrix of the environment
        """
        not_walls = self.__compute_walls_from_transition_matrix(action, transition_matrix)
        update = False

        for w in self.states_space.values():
            if np.array_equal(not_walls, w):
                # TODO: update policy using idk what but consider rewards
                update = True
                break
        if not update:
            self.states_space[len(self.states_space)] = not_walls
            self.policy[len(self.policy)] = not_walls / np.sum(not_walls)

    def get_state_number(self, action: int, transition_matrix: np.ndarray):
        not_walls = self.__compute_walls_from_transition_matrix(action, transition_matrix)
        for state_number, walls in self.states_space.items():
            if np.array_equal(not_walls, walls):
                return state_number
    
    def compute_next_action(self, action: int, transition_matrix: np.ndarray):
        self.__update_states_space(action, transition_matrix)
        state_number = self.get_state_number(action, transition_matrix)
        action_agent_pov = np.random.choice(ACTIONS_LIST, p=self.policy[state_number])
        
        # map action from agent's point of view to environment's point of view
        return WALLS_MAP[action][action_agent_pov]
    
    def reset_at_initial_state(self, transition_matrix_initial_state: np.ndarray):
        self.__compute_fitizial_first_action(transition_matrix_initial_state)
        self.update_policy(action=self.fittizial_first_action, 
            transition_matrix=transition_matrix_initial_state)

    def __update_states_space(self, action:int, transition_matrix:np.ndarray):
        """
        Update states space with new state if it is not already present in the states space

        Args:
            action (int): action taken by the agent
            transition_matrix (np.ndarray): transition matrix of the environment
        """
        not_walls = self.__compute_walls_from_transition_matrix(action, transition_matrix)
        update = True
        for w in self.states_space.values():
            if np.array_equal(not_walls, w):
                update = False
                break
        if update:
            self.states_space[len(self.states_space)] = not_walls
            self.policy[len(self.policy)] = not_walls / np.sum(not_walls)
    
    def __compute_walls_from_transition_matrix(self, action:int, transition_matrix:np.ndarray):
        # compute walls from the agent's point of view: 0 if there is a wall, 1 otherwise
        # wall 0: agent's left
        # wall 1: agent's below
        # wall 2: agent's right
        # wall 3: agent's above
        env_not_walls = np.sum(transition_matrix, axis=0)
        return env_not_walls[WALLS_MAP[action]]
    
    def __compute_fitizial_first_action(self, transition_matrix_initial_state: np.ndarray):
        # compute a fitizial action to possibly arrive in initial state
        no_walls = np.sum(transition_matrix_initial_state, axis=0)
        first_action_map = {
            0: 1,
            1: 0,
            2: 3,
            3: 2
        }
        self.fittizial_first_action = first_action_map[np.random.choice(np.where(no_walls != 0)[0])]