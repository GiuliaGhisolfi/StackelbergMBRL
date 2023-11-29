import json
import numpy as np
from src.agents.agent import Agent
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]
WALLS_MAP = {
    0: np.array([2, 0, 3, 1]), # up action:    right, down, left, up
    1: np.array([3, 1, 2, 0]), # down action:  left, up, right, down
    2: np.array([1, 2, 0, 3]), # left action:  down, left, up, right
    3: np.array([0, 3, 1, 2])  # right action: up, right, down, left
}

class StackelbergAgent(Agent):
    def __init__(self, initial_state_coord, policy_path, states_space_path, transition_matrix_initial_state):
        super().__init__(initial_state_coord)
    
        self.__read_policy(policy_path, states_space_path)
        self.__compute_fitizial_first_action(transition_matrix_initial_state)

        self.agent_state_coord = initial_state_coord
        self.path = [self.agent_state_coord]
    
    def take_action(self, action:int, transition_matrix:np.ndarray):
        not_walls = self.compute_walls_from_transition_matrix(action, transition_matrix)
        not_walls_index = [key for key, value in self.states_space.items()
            if np.equal(value, not_walls).all()]

        if len(not_walls_index) == 1:
            action_agent_pov = np.random.choice(ACTIONS_LIST, p=self.policy[not_walls_index[0]]) # relative action
        else:
            action_agent_pov = np.random.choice(ACTIONS_LIST, p=not_walls / np.sum(not_walls)) # relative action
        
        # map action from agent's point of view to environment's point of view
        return WALLS_MAP[action][action_agent_pov]

    def update_agent_parameters(self, next_state_coord):
        # update current state and path
        self.agent_state_coord = next_state_coord
        self.path.append(next_state_coord)
    
    def __read_policy(self, policy_path, states_space_path):
        # import policy from json file
        with open(policy_path, 'r') as policy_file:
            policy = json.load(policy_file)
        self.policy = [np.array(arr) for arr in policy]

        # import states space from json file
        with open(states_space_path, 'r') as states_space_file:
            states_space = json.load(states_space_file)
        self.states_space = {int(key): np.array(value) for key, value in states_space.items()}
    
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
    
    def __compute_fitizial_first_action(self, transition_matrix_initial_state:np.ndarray):
        # compute a fitizial action to possibly arrive in initial state
        no_walls = np.sum(transition_matrix_initial_state, axis=0)
        first_action_map = {
            0: 1,
            1: 0,
            2: 3,
            3: 2
        }
        self.fittizial_first_action = first_action_map[np.random.choice(np.where(no_walls != 0)[0])]


