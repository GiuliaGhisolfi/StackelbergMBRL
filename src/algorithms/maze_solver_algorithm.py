import numpy as np
from src.agents.model_agent import ModelAgent
from src.agents.policy_agent import PolicyAgent
from src.environment.environment import Environment
from src.algorithms.utils import softmax_gradient, save_parameters


ACTIONS_MAP = {
    0: [1, 3, 0, 2], 
    1: [3, 1, 2, 0], 
    2: [2, 0, 1, 3], 
    3: [0, 2, 3, 1]
    } # actions map from agent's pov
WALLS_MAP = {
    0: np.array([2, 0, 3, 1]), # up action:    right, down, left, up
    1: np.array([3, 1, 2, 0]), # down action:  left, up, right, down
    2: np.array([1, 2, 0, 3]), # left action:  down, left, up, right
    3: np.array([0, 3, 1, 2])  # right action: up, right, down, left
}

class MazeSolverAlgorithm():
    
    def __init__(self, algorithm, learning_rate, n_environments, max_iterations_per_environment, n_episodes_per_iteration, 
        max_epochs_per_episode, maze_width, maze_height, alpha, gamma, epsilon):

        self.algorithm = algorithm # 'PAL' or 'MAL'
        self.n_environments = n_environments
        self.max_iterations_per_environment = max_iterations_per_environment
        self.n_episodes_per_iteration = n_episodes_per_iteration
        self.max_epochs_per_episode = max_epochs_per_episode
        self.lr = learning_rate # learning rate for policy improvement
        self.alpha = alpha # alpha, learning rate for value function update
        self.gamma = gamma # discount factor to compute expected cumulative reward
        self.epsilon = epsilon # epsilon greedy parameter

        parameters_dict = {
            'algorithm': self.algorithm,
            'n_environments': self.n_environments,
            'max_iterations_per_environment': self.max_iterations_per_environment,
            'n_episodes_per_iteration': self.n_episodes_per_iteration,
            'max_epochs_per_episode': self.max_epochs_per_episode,
            'learning_rate': self.lr,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon
            }
        save_parameters(parameters_dict, algorithm=self.algorithm) # save parameters in json file

        # initalize environment
        self.env = Environment(
            maze_width=maze_width,
            maze_height=maze_height
            )

        # initialize model and policy agents
        self.model_agent = ModelAgent(
            gamma=gamma,
            initial_state_coord=self.env.initial_state_coord,
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :]
            )
        self.policy_agent = PolicyAgent(
            gamma=gamma,
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],          
            epsilon=epsilon
            )
        print('Agents initialized')
        
        self.map_state_model_to_policy = dict() # map function := {(x,y): state_policy}
        self.map_state_model_to_policy[(self.env.initial_state_coord,
            self.policy_agent.fittizial_first_action)] = 0 # initial state
    
    def reset_at_initial_state(self):
        # reset agent's position in the environment
        self.env.reset()

        # reset agent state at initial state
        self.policy_agent.reset_at_initial_state(
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])
    
    def executing_policy(self):
        """
        Execute policy in the environment to collect data

        Returns:
            episode (list): list of tuples (state, action, reward, next_state)
        """
        current_state = self.env.state
        current_state_coord = self.env.coordinates_from_state(self.env.state)
        action = self.policy_agent.fittizial_first_action
        episode = [] # init

        for _ in range(self.max_epochs_per_episode):
            # select action from policy
            action = self.policy_agent.compute_next_action(action=action, 
                        transition_matrix=self.env.p[:, current_state, :])

            # compute step in environment
            next_state, reward, _, _, _ = self.env.step(action) # next_state: numeric

            # save step in episode
            episode.append((current_state_coord, action, reward, self.env.coordinates_from_state(next_state)))

            # update current state
            current_state = next_state
            current_state_coord = self.env.coordinates_from_state(next_state)
            
            # check if terminal state is reached
            if current_state == self.env.terminal_state:
                print('Terminal state reached')
                break
            
        return episode