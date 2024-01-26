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
    def __init__(self, gamma:float, transition_matrix_initial_state:np.ndarray, 
    compute_action_policy:str='epsilon_greedy', epsilon:float=0.1):
        self.policy = None # policy[state number] = probability distribution over actions
        self.states_space = dict() # {state number: walls[-,-,-,-]}

        self.quality_function = dict() # Q := {state number: actions values (np.darray)}
        self.quality_function[0] = np.zeros(N_ACTIONS)

        self.gamma = gamma # discount factor
        self.compute_action_policy = compute_action_policy # epsilon_greedy or uniform_initialization
        self.epsilon = epsilon

        self.compute_fitizial_first_action(transition_matrix_initial_state)
    
    def reset_at_initial_state(self, transition_matrix_initial_state:np.ndarray):
        self.compute_fitizial_first_action(transition_matrix_initial_state)
    
    def compute_next_action(self, action:int, transition_matrix:np.ndarray):
        self.update_states_space(action, transition_matrix)
        state_number = self.get_state_number(action, transition_matrix)
        action_agent_pov = np.random.choice(ACTIONS_LIST, p=self.policy[state_number, :]) # relative action
        
        # map action from agent's point of view to environment's point of view
        return WALLS_MAP[action][action_agent_pov]
    
    def get_state_number(self, action:int, transition_matrix:np.ndarray):
        not_walls = self.compute_walls_from_transition_matrix(action, transition_matrix)
        for state_number, walls in self.states_space.items():
            if np.array_equal(not_walls, walls):
                return state_number

    def update_states_space(self, action:int, transition_matrix:np.ndarray):
        """
        Update states space with new state if it is not already present in the states space

        Args:
            action (int): action taken by the agent
            transition_matrix (np.ndarray): transition matrix of the environment
        """
        not_walls = self.compute_walls_from_transition_matrix(action, transition_matrix)

        # add state to states space if it isn't already in
        if len([key for key, value in self.states_space.items() if np.equal(value, not_walls).all()]) < 1:
            self.states_space[len(self.states_space)] = not_walls
            # update policy
            if np.sum(not_walls) > 1:
                policy = not_walls * self.epsilon / (np.sum(not_walls) - 1)
                policy[np.where(not_walls == 1)[0][0]] = 1 - self.epsilon # epsilon greedy
            else:
                policy = not_walls.astype(float)
            self.policy = np.concatenate((self.policy, policy.reshape(1, -1)), axis=0
                ) if self.policy is not None else policy.reshape(1, -1)
    
    def compute_walls_from_transition_matrix(self, action:int, transition_matrix:np.ndarray):
        # compute walls from the agent's point of view: 0 if there is a wall, 1 otherwise
        # wall 0: agent's left
        # wall 1: agent's below
        # wall 2: agent's right
        # wall 3: agent's above, always 1
        if len(transition_matrix.shape) > 1:
            env_not_walls = np.sum(transition_matrix, axis=0)
        else:
            env_not_walls = transition_matrix

        return np.array([0 if i==0 else 1 for i in env_not_walls[WALLS_MAP[action]]])
    
    def compute_fitizial_first_action(self, transition_matrix_initial_state:np.ndarray):
        # compute a fitizial action to possibly arrive in initial state
        no_walls = np.sum(transition_matrix_initial_state, axis=0)
        first_action_map = {
            0: 1,
            1: 0,
            2: 3,
            3: 2
        }
        self.fittizial_first_action = first_action_map[np.random.choice(np.where(no_walls != 0)[0])]
        self.update_states_space(action=self.fittizial_first_action, 
            transition_matrix=transition_matrix_initial_state)