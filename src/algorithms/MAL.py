import numpy as np
import random
from src.algorithms.maze_solver import MazeSolver
from src.algorithms.utils import save_metrics, save_policy, check_stackelberg_nash_equilibrium

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]
WALLS_MAP = {
    0: np.array([2, 0, 3, 1]), # up action:    right, down, left, up
    1: np.array([3, 1, 2, 0]), # down action:  left, up, right, down
    2: np.array([1, 2, 0, 3]), # left action:  down, left, up, right
    3: np.array([0, 3, 1, 2])  # right action: up, right, down, left
}

class MAL(MazeSolver):
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
                    initial_state_coord=self.env.initial_state_coord,
                    transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])

                # reset policy agent at initial state
                self.policy_agent.reset_at_initial_state(
                    transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])
    
    def train_loop_for_the_environment(self, environment_number):
        # train loop for the environment
        for i in range(self.max_iterations_per_environment):
            print(f'\nIteration {i+1}/{self.max_iterations_per_environment} for environment {environment_number}')

            # optimize policy
            if (len(self.model_agent.quality_function) > 1 or 
            np.sum(list(self.model_agent.quality_function.values()))):
                policy, policy_cost_function = self.optimize_policy()
                self.policy_agent.policy = policy
            else:
                policy_cost_function = 0
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

            for j, episode in enumerate(data_buffer):
                quality_function = self.improve_model(episode=episode)
                local_model_loss = self.compute_model_loss(quality_function=quality_function)

                if local_model_loss < model_loss:
                    model_loss = local_model_loss
                    optimal_quality_function = quality_function
                    optimal_model = j

                model_loss_list.append(-local_model_loss)
            self.model_agent.quality_function = optimal_quality_function
            self.policy_agent.quality_function = self.compute_quality_function_policy(
                quality_function=optimal_quality_function)
            print('Model optimized')

            # stopping criterion
            if i == 0:
                print('First iteration, no stopping criterion')
                metrics_dict = {
                    'algorithm': 'MAL',
                    'environment_number': environment_number,
                    'iteration_number': i+1,
                    'mse': model_loss_list,
                    'cost_function': policy_cost_function,
                    'best_model': optimal_model,
                    'nash_equilibrium_found': None,
                    }
                save_metrics(metrics_dict, model_values_function=self.model_agent.quality_function, 
                    environment_number=environment_number, iteration_number=i+1, algorithm='MAL')
            else:
                if check_stackelberg_nash_equilibrium(leader_payoffs=model_loss_list, 
                    follower_payoffs=[policy_cost_function], equilibrium_find=(optimal_model, 0)):
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
        Returns:
            policy (np.ndarray): policy to execute in the environment
        """
        # compute policy from model's quality function
        policy = self.policy_agent.policy.copy()
        quality_function = self.model_agent.quality_function.copy()

        # update policy maximaizing model reward
        quality_function_policy = self.compute_quality_function_policy(quality_function)
        for state, not_walls in self.policy_agent.states_space.items():
            if np.sum(not_walls) > 1:
                policy[state]= not_walls * self.epsilon / (np.sum(not_walls) - 1)
                max_value = max(quality_function_policy[state]*not_walls)
                idx = np.where(quality_function_policy[state] == max_value)[0] #FIXME
                policy[state][idx[0]] = 1 - self.epsilon # epsilon greedy
            else:
                policy[state] = not_walls.astype(float)
  
        # execute policy in the model to collect reward
        rewards = []
        state = random.choice(list(self.model_agent.states_space)) # random initial state

        for _ in range(int(self.max_epochs_per_episode/10)):
            # state, action, reward, next state
            action = np.random.choice(np.where(
                self.model_agent.quality_function[state] != 0)[0])
            rewards.append(self.model_agent.reward_function[(state, action)])
            state = self.model_agent.next_state_function[(state, action)] # next state
            if rewards[-1] == 0:
                break # terminal state reached
        
        # compute cost function
        value_function_last_state = np.sum(quality_function_policy[
            self.map_state_model_to_policy[state, action]] * policy)
        cumulative_reward = self.compute_cumulative_reward(rewards)
        cost_function = cumulative_reward + value_function_last_state

        return policy, cost_function
    
    def compute_cumulative_reward(self, rewards):
        T = len(rewards) # number of steps to reach terminal state from current state
        if T == 1:
            return rewards[0]  # return the reward of the final step
        else:
            return rewards[0] + self.gamma * self.compute_cumulative_reward(rewards[1:])  # recursive call
    
    def compute_quality_function_policy(self, quality_function):
        # map function := {((x,y), previous_action): state_policy}
        # Q (x,y): actions values (np.darray)
        quality_function_policy = self.policy_agent.quality_function.copy()
        for i in range(len(self.policy_agent.states_space)-len(quality_function_policy)+1):
            quality_function_policy[i] = np.zeros(N_ACTIONS) 
        
        # update quality function policy
        for state, actions_values in quality_function.items():
            for previous_action in ACTIONS_LIST:
                try:
                    quality_function_policy[self.map_state_model_to_policy[(state,
                        previous_action)]] += actions_values
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
            # update states space and model's parameters
            self.update_states_space(state_coord=state, previous_action=previous_action)
            self.model_agent.update_next_state_function(state, action, next_state)
            self.model_agent.update_reward_function(state, action, reward)

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
        Update policy agent's states space with new state if it is not already present in the states space

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
            np.array(list(quality_function.values())))
