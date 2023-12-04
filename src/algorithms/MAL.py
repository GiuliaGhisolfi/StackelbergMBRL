import numpy as np
from src.algorithms.maze_solver_algorithm import MazeSolverAlgorithm
from src.algorithms.utils import save_metrics, save_policy, check_stackelberg_nash_equilibrium_MAL

N_ACTIONS = 4
ACTION_LIST = [0, 1, 2, 3] # [up, down, left, right]

class MAL(MazeSolverAlgorithm):
    # MAL: Model As Leader Algorithm

    def __init__(self, learning_rate, n_environments, max_iterations_per_environment, n_episodes_per_iteration, 
        max_epochs_per_episode, maze_width, maze_height, alpha, gamma, epsilon):

        super().__init__(algorithm='MAL', learning_rate=learning_rate, n_environments=n_environments, 
            max_iterations_per_environment=max_iterations_per_environment, n_episodes_per_iteration=n_episodes_per_iteration, 
            max_epochs_per_episode=max_epochs_per_episode, maze_width=maze_width, maze_height=maze_height, 
            alpha=alpha, gamma=gamma, epsilon=epsilon)
    
    def train(self):
        # train loop over n_environments environments
        print('Training started')
        for i in range(self.n_environments):
            print(f'\nTraining on environment {i+1}/{self.n_environments}')
            # train loop for the environment
            self.train_loop_for_the_environment(environment_number=i+1)

            # checkpoint: save policy, policy states space and metrics in json file
            save_policy(policy=self.policy_agent.policy, states_space=self.policy_agent.states_space, 
                algorithm='MAL', environment_number=i)

            if i < self.n_environments - 1:
                # reset and initialize new environment and model agent
                self.env.reset_environment()
                self.model_agent.reset_agent(
                    initial_state_coord=self.env.initial_state_coord)

                # reset policy agent at initial state
                self.policy_agent.reset_at_initial_state(
                    transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])
    
    def train_loop_for_the_environment(self, environment_number):
        # train loop for the environment
        for i in range(self.max_iterations_per_environment):
            print(f'\nIteration {i+1}/{self.max_iterations_per_environment} for environment {environment_number}')

            # optimize policy
            policy = self.optimize_policy()
            policy_cost_function = self.compute_policy_cost_function(policy=policy,
                quality_function=self.model_agent.quality_function)
            self.policy_agent.policy = policy
            print('Policy optimized')

            # collect data executing policy in the environment
            data_buffer = [] # list of episodes, episode: list of tuples (state, action, reward, next_state)
            for _ in range(self.n_episodes_per_iteration):
                episode = self.executing_policy()
                data_buffer.append(episode)
                self.reset_at_initial_state() # reset environment and agent state
            print(f'Collected {len(data_buffer)} episodes')

            # improve model
            model_loss = np.inf
            model_loss_list = []
            optimal_model = -1

            for i, episode in enumerate(data_buffer):
                quality_function = self.improve_model(episode=episode)
                local_model_loss = self.compute_model_loss(quality_function=quality_function)

                if local_model_loss < model_loss:
                    model_loss = local_model_loss
                    optimal_quality_function = quality_function
                    optimal_model = i

                model_loss_list.append(-local_model_loss)
            self.model_agent.quality_function = optimal_quality_function
            print('Model optimized')

            # stopping criterion
            follower_payoffs = policy_cost_function * np.ones(len(model_loss_list))
            if check_stackelberg_nash_equilibrium_MAL(leader_payoffs=model_loss_list, 
                follower_payoffs=follower_payoffs, target_value=optimal_model):
                print('Stackelberg-Nash equilibrium reached')
                metrics_dict = {
                    'algorithm': 'MAL',
                    'environment_number': environment_number,
                    'iteration_number': i+1,
                    'mse': model_loss_list,
                    'cost_function': policy_cost_function,
                    'best_model': optimal_model,
                    'nash_equilibrium_found': True,
                    }
                save_metrics(metrics_dict, model_values_function=self.model_agent.quality_function, 
                    environment_number=environment_number, iteration_number=i+1, algorithm='MAL')
                break
            else:
                print('Stackelberg-Nash equilibrium not reached')
                metrics_dict = {
                    'algorithm': 'MAL',
                    'environment_number': environment_number,
                    'iteration_number': i+1,
                    'mse': model_loss_list,
                    'cost_function': policy_cost_function,
                    'best_model': optimal_model,
                    'nash_equilibrium_found': False,
                    }
                save_metrics(metrics_dict, model_values_function=self.model_agent.quality_function, 
                    environment_number=environment_number, iteration_number=i+1, algorithm='MAL')
    
    ###### policy optimization ######
    def optimize_policy(self):
        """
        Optimize policy using gradient ascent

        Args:
            quality_function (dict): model's quality function
        
        Returns:
            policy (np.ndarray): policy to execute in the environment
        """
        # compute policy from model's quality function
        policy = self.policy_agent.policy.copy()
        quality_function = self.model_agent.quality_function.copy()

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
        value_function_approx = np.sum(np.concatenate((quality_function_policy, np.zeros((
            len(self.policy_agent.states_space)-len(quality_function_policy), N_ACTIONS))), axis=0)
            * policy, axis=1).reshape(-1,1)
        return quality_function_policy - np.ones(quality_function_policy.shape) * value_function_approx
    
    def compute_quality_function_policy(self, quality_function):
        # map function := {((x,y), previous_action): state_policy}
        # Q (x,y): actions values (np.darray)
        quality_function_policy = np.zeros((len(self.policy_agent.states_space), N_ACTIONS))
        for state, actions_values in quality_function.items():
            for previous_action in ACTION_LIST:
                try:
                    quality_function_policy[self.map_state_model_to_policy[(state,
                        previous_action)]] = actions_values
                except:
                    pass
        
        return quality_function_policy

    ####### model improvement ######
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
            self.update_states_space(state_coord=state, previous_action=previous_action)

            if next_state not in quality_function.keys():
                # initialize quality function at next state
                quality_function[next_state] = np.zeros(N_ACTIONS)
            
            # Q-Learning update
            quality_function[state][action] += self.alpha * (reward + 
                self.gamma * np.max(quality_function[next_state]) - quality_function[state][action])
            
            previous_action = action
        
        return quality_function
    
    def update_states_space(self, state_coord, previous_action):
        """
        Update states space with new state if it is not already present in the states space

        Args:
            state_coord (tuple): coordinates of the state
            previous_action (int): action taken by the agent to reach the state
        """
        if (state_coord, previous_action) not in self.map_state_model_to_policy.keys():
            self.policy_agent.update_states_space(action=previous_action, 
                transition_matrix=self.env.p[:,self.env.state_from_coordinates(
                state_coord[0], state_coord[1]), :])

            not_walls = self.policy_agent.compute_walls_from_transition_matrix(action=previous_action,
                transition_matrix=self.env.p[:,self.env.state_from_coordinates(state_coord[0], state_coord[1]), :])
            state_number = [key for key, value in self.policy_agent.states_space.items() if 
                np.equal(value, not_walls).all()][0]

            self.map_state_model_to_policy[(state_coord, previous_action)] = state_number
    
    def compute_model_loss(self, quality_function):
        def mse(y_true, y_pred):
            return np.mean(np.power(y_true - y_pred, 2))
        
        return mse(np.concatenate((np.array(list(self.model_agent.quality_function.values())),                
            np.zeros((len(quality_function)-len(self.model_agent.quality_function), N_ACTIONS))), axis=0),
            np.array(list(quality_function.values()))[:len(self.model_agent.quality_function)])
