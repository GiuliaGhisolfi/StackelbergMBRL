import numpy as np
import copy
import random
import time
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

def softmax(x):
    exps = np.exp(x - np.max(x)) # subtracting the maximum avoids overflow
    return exps / np.sum(exps)

def softmax_prime(x):
    f = softmax(x)
    return np.diag(np.diagflat(f) - np.outer(f, f))

class PAL(MazeSolver):
    # PAL: Policy As Leader Algorithm

    def __init__(self, learning_rate, n_environments, max_iterations_per_environment, n_episodes_per_iteration, 
        max_epochs_per_episode, maze_width, maze_height, alpha, gamma, epsilon, verbose):

        super().__init__(algorithm='PAL', learning_rate=learning_rate, n_environments=n_environments, 
            max_iterations_per_environment=max_iterations_per_environment, n_episodes_per_iteration=n_episodes_per_iteration, 
            max_epochs_per_episode=max_epochs_per_episode, maze_width=maze_width, maze_height=maze_height, 
            alpha=alpha, gamma=gamma, epsilon=epsilon, verbose=verbose)
        self.verbose = verbose
    
    def train(self):
        # train loop over n_environments environments
        print('Training started')
        start_time = time.time()
        for i in range(self.n_environments):
            print(f'\nTraining on environment {i+1}/{self.n_environments}')
            # train loop for the environment
            self.train_loop_for_the_environment(environment_number=i+1)

            # checkpoint: save policy, policy states space and metrics in json file
            save_policy(policy=self.policy_agent.policy, states_space=self.policy_agent.states_space, 
                algorithm='PAL', environment_number=i)

            if i < self.n_environments - 1:
                # reset and initialize new environment and model agent
                self.env.reset_environment()
                self.model_agent.reset_agent(
                    initial_state_coord=self.env.initial_state_coord,
                    transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])

                # reset policy agent at initial state
                self.policy_agent.reset_at_initial_state(
                    transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])
        print(f'End training after {time.time() - start_time} seconds')
    
    def train_loop_for_the_environment(self, environment_number):
        # train loop for the environment
        model_losses = []
        policy_cost_functions = []
        for i in range(self.max_iterations_per_environment):
            if self.verbose:
                print(f'\nIteration {i+1}/{self.max_iterations_per_environment} for environment {environment_number}')

            # collect data executing policy in the environment
            data_buffer = [] # list of episodes, episode: list of tuples (state, action, reward, next_state)
            for _ in range(self.n_episodes_per_iteration):
                episode = self.executing_policy()
                data_buffer.append(episode)
                self.reset_at_initial_state() # reset environment and agent state
            if self.verbose:
                print(f'Collected {len(data_buffer)} episodes')

            # optimize model and improve policy parameters
            model_loss = np.inf
            model_loss_list = []
            optimal_model = -1

            policy_cost_function = -np.inf
            policy_cost_function_list = []
            optimal_policy = -1

            for j, episode in enumerate(data_buffer):
                # optimize model
                quality_function, next_state_function, reward_function = self.optimize_model(episode=episode)
                local_model_loss = self.compute_model_loss(quality_function=quality_function)

                if local_model_loss < model_loss:
                    model_loss = local_model_loss
                    optimal_quality_function = quality_function
                    optimal_next_state_function = next_state_function
                    optimal_reward_function = reward_function
                    optimal_model = j

                model_loss_list.append(-local_model_loss)

                # improve policy
                policy, local_policy_cost_function = self.improve_policy(
                    model_agent_quality_function=quality_function, 
                    model_agent_next_state_function=next_state_function, 
                    model_agent_reward_function=reward_function)  
                
                if local_policy_cost_function > policy_cost_function:
                    policy_cost_function = local_policy_cost_function
                    optimal_policy = policy 
                    optimal_policy_agent = j

                policy_cost_function_list.append(local_policy_cost_function)

            # update model and policy agent
            self.model_agent.quality_function = optimal_quality_function
            self.model_agent.next_state_function = optimal_next_state_function
            self.model_agent.reward_function = optimal_reward_function

            self.policy_agent.policy = optimal_policy

            self.policy_agent.quality_function = self.compute_quality_function_policy(
                quality_function=optimal_quality_function)
            if self.verbose:
                print('Model optimized and policy improved')

            # stopping criterion
            if check_stackelberg_nash_equilibrium(leader_payoffs=model_loss_list, 
                follower_payoffs=policy_cost_function_list, equilibrium_find=(optimal_model, optimal_policy_agent)):
                if self.verbose:
                    print('Stackelberg-Nash equilibrium reached')
                break
            else:
                if self.verbose:
                    print('Stackelberg-Nash equilibrium not reached')
            
            model_losses.append(model_loss_list[optimal_model])
            policy_cost_functions.append(policy_cost_function_list[optimal_policy_agent])
        
        metrics_dict = {
            'algorithm': 'PAL',
            'environment_number': environment_number,
            'iteration_done': i,
            'mse': model_losses,
            'cost_function': policy_cost_functions,
            }           
        save_metrics(metrics_dict, environment_number=environment_number, algorithm='PAL')
        
    ###### policy optimization ######
    def improve_policy(self, model_agent_quality_function, model_agent_next_state_function, 
        model_agent_reward_function):
        """
        Returns:
            policy (np.ndarray): policy to execute in the environment
        """
        # compute policy from model's quality function
        policy = copy.deepcopy(self.policy_agent.policy)
        quality_function = copy.deepcopy(model_agent_quality_function)

        # update policy maximaizing model reward
        quality_function_policy = self.compute_quality_function_policy(quality_function)
        for state, not_walls in self.policy_agent.states_space.items():
            if np.sum(not_walls) > 1:
                value = quality_function_policy[state]*not_walls
                value = [-np.inf if abs(value[i]) == 0 else value[i] for i in range(4)]
                idx = np.where(value == max(value))[0]
                if len(idx) > 0:
                    policy[state] = not_walls * self.epsilon / (np.sum(not_walls) - 1)
                    policy[state][idx[0]] = 1 - self.epsilon # epsilon greedy
                else:
                    policy[state] = not_walls * self.epsilon / (np.sum(not_walls))
            else:
                policy[state] = not_walls.astype(float)
  
        # execute policy in the model to collect reward
        rewards = []
        policy_states = []
        state = random.choice(list(model_agent_quality_function)) # random initial state
        for previous_action in range(4):
            if (state, previous_action) in self.map_state_model_to_policy.keys():
                break

        for _ in range(int(self.max_epochs_per_episode/2)):
            # state, action, reward, next state
            action = np.random.choice(np.where(
                model_agent_quality_function[state] != 0)[0])
            rewards.append(model_agent_reward_function[(state, action)])
            policy_states.append(self.map_state_model_to_policy[state, previous_action])
            state = model_agent_next_state_function[(state, action)] # next state

            # stopping criteria
            if rewards[-1] == 0 or len(np.where(model_agent_quality_function[state] != 0)[0]) == 0:
                break # terminal state reached

            previous_action = action
        
        # improve policy

        # initalize policy parameters
        theta = np.random.uniform(0, 1, size=(len(self.policy_agent.states_space), N_ACTIONS))

        # reinforce Monte Carlo
        for r in range(len(rewards)):
            # policy paramether update
            cumulative_reward = self.compute_cumulative_reward(rewards[r:])
            gradient = softmax_prime(theta[policy_states[r]]) / softmax(theta[policy_states[r]])
            theta[policy_states[r]] = theta[policy_states[r]] + self.alpha * cumulative_reward * gradient
        
        # compute policy from policy parameters
        policy = softmax(theta)
        mask = np.array(list(self.policy_agent.states_space.values()))
        policy = policy * mask # remove walls from policy
        policy = policy / np.sum(policy, axis=1, keepdims=True) # normalize policy
        
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
        quality_function_policy = copy.deepcopy(self.policy_agent.quality_function)

        delta_len = len(quality_function_policy)
        for i in range(len(self.policy_agent.states_space)-delta_len):
            quality_function_policy[delta_len+i] = np.zeros(N_ACTIONS) 
        
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
    def optimize_model(self, episode):
        """
        Improve model's parameters from experience using episode

        Args:
            episode (list): list of tuples (state, action, reward, next_state)
        
        Returns:
            quality_function (dict): updated model agents quality function
        """
        # create a dict 
        quality_function = copy.deepcopy(self.model_agent.quality_function)
        next_state_function = copy.deepcopy(self.model_agent.next_state_function)
        reward_function = copy.deepcopy(self.model_agent.reward_function)
        previous_action = self.policy_agent.fittizial_first_action
        
        for state, action, reward, next_state in episode:
            # update states space and model's parameters
            self.update_states_space(state_coord=state, previous_action=previous_action)
            next_state_function[(state, action)] = next_state
            if (state, action) not in reward_function.keys():
                reward_function[(state, action)] = reward
            else:
                reward_function[(state, action)] += reward

            if next_state not in quality_function.keys():
                # initialize quality function at next state
                quality_function[next_state] = np.zeros(N_ACTIONS)
            
            # Q-Learning update
            quality_function[state][action] += self.alpha * (reward + 
                self.gamma * np.max(quality_function[next_state]) - quality_function[state][action])
                        
            previous_action = action

            c = len(np.nonzero(np.array(list(quality_function.values()))))
            d = len(reward_function)
            e = len(next_state_function)
        self.update_states_space(state_coord=next_state, previous_action=previous_action) # last state
        
        return quality_function, next_state_function, reward_function
    
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
