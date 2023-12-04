import numpy as np
from src.agents.model_agent import ModelAgent
from src.agents.policy_agent import PolicyAgent
from src.environment.environment import Environment
from src.algorithms.utils import softmax_gradient, save_parameters

N_ACTIONS = 4
ACTION_LIST = [0, 1, 2, 3] # [up, down, left, right]
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
            initial_state_coord=self.env.initial_state_coord
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
    
    def optimize_policy(self, quality_function):
        """
        Optimize policy using gradient ascent

        Args:
            quality_function (dict): model's quality function
        
        Returns:
            policy (np.ndarray): policy to execute in the environment
        """
        # compute policy from model's quality function
        policy = self.policy_agent.policy.copy()

        # compute policy gradient
        policy_gradient = self.compute_policy_gradient(policy=policy, 
            quality_function=quality_function)

        # update policy
        policy += self.lr * policy_gradient

        return policy
    
    def compute_policy_gradient(self, policy, quality_function):
        # compute cost function gradient from quality function
        return np.nan_to_num(np.log(policy+1e-10) * self.compute_advantege_function(
            quality_function, policy)) # avoid log(0)

    def compute_policy_cost_function(self, policy, quality_function):
        return np.sum(policy * self.compute_advantege_function(quality_function, policy))
    
    def compute_advantege_function(self, quality_function, policy):
        # compute advantage function from quality function
        quality_function_policy = self.compute_quality_function_policy(quality_function)
        value_function_approx = np.sum(quality_function_policy * policy, axis=1)
        return np.max(quality_function_policy, axis=1) - value_function_approx
    
    def compute_quality_function_policy(self, quality_function):
        # map function := {(x,y): state_policy}
        # Q (x,y): actions values (np.darray)
        quality_function_policy = np.zeros((N_ACTIONS, len(self.policy_agent.states_space)))
        for state, actions_values in quality_function.items():
            for previous_action in ACTION_LIST:
                try:
                    quality_function_policy[self.map_state_model_to_policy[(state,
                        previous_action)]] = actions_values
                except:
                    pass
        
        return quality_function_policy

    def improve_model(self, episode):
        """
        Improve model's parameters from experience using episode

        Args:
            episode (list): list of tuples (state, action, reward, next_state)
        
        Returns:
            quality_function (dict): updated model agents quality function
        """
        # create a dict 
        quality_function = self.model_agent.quality_function.copy()
        previous_action = self.policy_agent.fittizial_first_action
        
        for state, action, reward, next_state in episode:
            # update states space
            self.update_states_space(state_coord=state, previus_action=previous_action)

            if next_state not in quality_function.keys():
                # initialize quality function at next state
                quality_function[next_state] = np.zeros(N_ACTIONS)
            
            # Q-Learning update
            quality_function[state][action] += self.alpha * (reward + 
                self.gamma * np.max(quality_function[next_state]) - quality_function[state][action])
            
            previous_action = action
        
        return quality_function
    
    def update_states_space(self, state_coord, previus_action):
        """
        Update states space with new state if it is not already present in the states space

        Args:
            state_coord (tuple): coordinates of the state
            previus_action (int): action taken by the agent to reach the state
        """
        if state_coord not in self.map_state_model_to_policy.keys():
            self.policy_agent.update_states_space(action=previus_action, 
            transition_matrix=self.env.p[:,self.env.state_from_coordinates(
            state_coord[0], state_coord[1]), :])

            self.map_state_model_to_policy[(state_coord, previus_action)] = len(self.map_state_model_to_policy)
    
    def compute_model_loss(self, quality_function):
        def mse(y_true, y_pred):
            return np.mean(np.power(y_true - y_pred, 2))
        
        return mse(np.array(list(self.model_agent.quality_function.values())), 
            np.array(list(quality_function.values()))[:len(self.model_agent.quality_function)])
