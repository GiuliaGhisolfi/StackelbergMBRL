import numpy as np
from src.algorithms.maze_solver import MazeSolver
from src.algorithms.utils import softmax_gradient, softmax, check_stackelberg_nash_equilibrium_PAL, \
    compute_model_loss, compute_actor_critic_objective, save_metrics, save_policy

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

class PAL(MazeSolver):
    # PAL: Policy As Leader Algorithm

    def __init__(self, learning_rate, n_environments, max_iterations_per_environment, n_episodes_per_iteration,
        max_epochs_per_episode, maze_width, maze_height, alpha, gamma, epsilon):
    
        super().__init__(algorithm='PAL', learning_rate=learning_rate, n_environments=n_environments, 
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
 
    def train_loop_for_the_environment(self, environment_number):
        nash_equilibrium_found = False
        for i in range(self.max_iterations_per_environment):
            print(f'\nIteration {i+1}/{self.max_iterations_per_environment} for environment {environment_number}')
            data_buffer = [] # list of episodes, each episode is a list of tuples (state, action, reward, next_state)

            # collect data executing policy in the environment
            for _ in range(self.n_episodes_per_iteration):
                episode = self.executing_policy()
                data_buffer.append(episode)
                self.reset_at_initial_state() # reset environment and agent state
            print(f'Collected {len(data_buffer)} episodes')
            
            # compute parameters to optimize model and improve policy
            p = np.array(self.model_agent.transition_distribuition) # transition probability matrix
            q_optimal, states_space_model, optimal_reward_function, optimal_next_state_function, \
                optimal_value_function, policy, states_space_policy = self.optimize_model_and_improve_policy(data_buffer, p)
            
            # build policy-specific model
            self.model_agent.transition_distribuition = q_optimal
            self.model_agent.states_space = states_space_model
            self.model_agent.next_state_function = optimal_next_state_function
            self.model_agent.reward_function = optimal_reward_function
            self.model_agent.values_function = optimal_value_function
            print(f'Model optimized: kl divergence = {self.kl_divergence[self.optimal_model]}')
            
            # improve policy
            self.policy_agent.policy = policy 
            self.policy_agent.states_space = states_space_policy
            print(f'Policy improved: cost function = {self.cost_function[self.optimal_model]}')

            # stopping criteria: stop in nash equilibrium
            if check_stackelberg_nash_equilibrium_PAL(leader_payoffs=self.kl_divergence,
                follower_payoffs=self.cost_function, leader_target_value=self.optimal_policy_agent, 
                follower_target_value=self.optimal_model):
                nash_equilibrium_found = True
                print(f'\nStackelberg nash equilibrium reached after {i+1} iterations \n')
                metrics_dict = {
                    'algorithm': 'PAL',
                    'environment_number': environment_number,
                    'iteration_number': i+1,
                    'kl_divergence': self.kl_divergence,
                    'cost_function': self.cost_function,
                    'best_model': self.optimal_model,
                    'nash_equilibrium_found': True,
                    }
                save_metrics(metrics_dict, model_values_function=self.model_agent.values_function, 
                    environment_number=environment_number, iteration_number=i+1, algorithm='PAL')
                break
            
            metrics_dict = {
                'algorithm': 'PAL',
                'environment_number': environment_number,
                'iteration_number': i+1,
                'kl_divergence': self.kl_divergence,
                'cost_function': self.cost_function,
                'best_model': self.optimal_model,
                'nash_equilibrium_found': False,
                }
            save_metrics(metrics_dict, model_values_function=self.model_agent.values_function, 
                    environment_number=environment_number, iteration_number=i+1, algorithm='PAL')

        if not nash_equilibrium_found:
            print(f'\nStackelberg nash equilibrium not reached after {self.max_iterations_per_environment} iterations \n')
        
    def optimize_model_and_improve_policy(self, data_buffer, p):
        """
        Build optimal model from experience using episode saved in data_buffer: 
        optimal model is the one that minimize the loss between current model and
        the model approximated from data_buffer

        Args:
            data_buffer (list): list of episodes, each episode is a list of tuples (state, action, reward, next_state)
            p (np.ndarray): transition probability matrix of the model
        """
        # model parameters
        loss = np.inf # model loss: kl divergence between model and data transition probability
        self.kl_divergence = [] # list of kl divergence between model and data
        self.optimal_model = -1 # optimal model index in data_buffer, to check if I'm in nash equilibrium

        q_optimal = np.empty((0, N_ACTIONS)) # init data transition probability as an empty array
        optimal_states_space_model = dict() # init optimal states space
        optimal_value_function = dict() # init optimal value function

        # policy parameters
        optimal_policy = []
        states_space_policy = dict()
        cost_function = -np.inf # cost function to maximize
        self.cost_function = [] # list of cost function computed from each episode

        # loop over episodes in data buffer
        for i, episode in enumerate(data_buffer):
            # compute model parameters for model optimization
            q, temp_states_space_model, temp_reward_function, temp_next_state_function, temp_value_function = \
                self.compute_transition_probability_from_episode(episode, p)

            local_loss = compute_model_loss(
                transition_probability_model=p,
                transition_probability_data=q)
            
            if local_loss < loss:
                loss = local_loss
                q_optimal = q
                self.optimal_model = i
                optimal_states_space_model = temp_states_space_model
                optimal_next_state_function = temp_next_state_function
                optimal_reward_function = temp_reward_function
                optimal_value_function = temp_value_function
            self.kl_divergence.append(local_loss) # in [0, +inf]

            # compute policy parameters for policy improvement
            if len(self.model_agent.values_function) > 1:
                model_values_function = np.array(list(self.model_agent.values_function.values()))
            else: # if only one state in the episode
                model_values_function = np.array(list(self.model_agent.values_function.values())).reshape(1, -1)
            
            advantages_model = np.array(list(temp_value_function.values())) - \
                np.concatenate((list(model_values_function), 
                np.zeros((len(temp_value_function) - len(self.model_agent.values_function), N_ACTIONS) )))
            
            temp_policy, temp_states_space_policy, advantages = self.improve_policy_from_episode(episode,
                advantages_model=advantages_model, states_space_model=temp_states_space_model)
            
            local_cost_function = compute_actor_critic_objective(temp_policy, advantages)

            if local_cost_function > cost_function:
                cost_function = local_cost_function
                self.optimal_policy_agent = i
                optimal_policy = temp_policy
                states_space_policy = temp_states_space_policy
            self.cost_function.append(local_cost_function)

        return (q_optimal, optimal_states_space_model, optimal_reward_function, optimal_next_state_function, 
            optimal_value_function, optimal_policy, states_space_policy)
    
    ###### model optimization ######
    def compute_transition_probability_from_episode(self, episode, p):
        """
        Compute transition probability from episode

        Args:
            episode (list): episode = [(state, action, reward, next_state), ...]

        Returns:
            q, temp_states_space, temp_reward_function, temp_next_state_function
        """
        states_model_space_dim = len(self.model_agent.states_space)

        temp_states_space = self.model_agent.states_space.copy()
        temp_next_state_function = self.model_agent.next_state_function.copy()
        temp_reward_function = self.model_agent.reward_function.copy()
        temp_value_function = self.model_agent.values_function.copy()

        # add new visited state to states space
        for state, action, reward, next_state in episode:
            if state not in temp_states_space.keys():
                temp_states_space[state] = states_model_space_dim 
                temp_value_function[states_model_space_dim] = np.zeros(N_ACTIONS) # init value function
                states_model_space_dim += 1

        # initialize transition probability matrix 
        q = np.zeros((states_model_space_dim, N_ACTIONS))
        q[:p.shape[0], :] = p
        q_weights = q.copy() # init transition probability weights
        mask = np.zeros((states_model_space_dim, N_ACTIONS)) # bool: possible actions from each state
        
        # update transition probability weights using actor-critic with TD(0) learning
        for state, action, reward, next_state in episode:
            # update reward function
            temp_reward_function[(temp_states_space[state], action)] = reward

            # update next state function
            if next_state in temp_states_space.keys():
                # not add last state visited in the episode to next state function
                temp_next_state_function[(temp_states_space[state], action)] = temp_states_space[next_state]

            # update value function using TD learning
            if next_state in temp_states_space.keys():
                td_error = reward + (self.gamma  * temp_value_function[temp_states_space[next_state]][action] - 
                    temp_value_function[temp_states_space[state]][action])
            else:
                td_error = reward - temp_value_function[temp_states_space[state]][action]

            temp_value_function[temp_states_space[state]][action] += self.alpha * td_error

            # update transition probability matrix
            gradient = softmax_gradient(policy=q_weights[temp_states_space[state], :], action=action)
            q_weights[temp_states_space[state], :] += temp_value_function[temp_states_space[state]] * gradient

            # update mask
            mask[temp_states_space[state], action] = 1

        q_weights = softmax(q_weights)
        q += self.lr * np.multiply(q_weights, mask) # apply mask to q_weights
        
        q /= np.sum(q, axis=1)[:, None]
        q[np.isnan(q)] = 0 # if 0/0, then 0

        return q, temp_states_space, temp_reward_function, temp_next_state_function, temp_value_function
    
    ###### policy improvement ######
    def improve_policy_from_episode(self, episode, advantages_model, states_space_model):
        temp_policy = self.policy_agent.policy.copy()
        temp_states_space_policy = self.policy_agent.states_space.copy()
        previous_action = self.policy_agent.fittizial_first_action
        advantages_policy = np.zeros((len(temp_states_space_policy), N_ACTIONS)) # cumulative temporal difference error
            
        for state, action, reward, next_state in episode:
            # transition probability matrix of the environment
            temp_policy, temp_states_space_policy, advantages_policy = self.update_policy(state, 
                temp_policy, temp_states_space_policy, action, previous_action, 
                advantages_model, states_space_model, advantages_policy)
            
            previous_action = action
        
        # policy as a distribution over actions
        temp_policy = softmax(temp_policy)

        mask = np.array(list(self.policy_agent.states_space.values()))
        temp_policy = np.multiply(temp_policy, mask) # apply mask to q_weights

        temp_policy = [temp_states_space_policy[i].astype(float).tolist() if np.sum(state_policy)==0 else 
            state_policy for i, state_policy in enumerate(temp_policy)]
        temp_policy /= np.sum(np.array(temp_policy), axis=1)[:, None] # normalize policy
        temp_policy[np.isnan(temp_policy)] = 0 # if 0/0, then 0        

        return temp_policy, temp_states_space_policy, advantages_policy
    
    def update_policy(self, state, policy:list, states_space_policy:dict, action:int, previous_action:int,
        advantages_model:list, states_space_model:dict, advantages_policy:np.ndarray):

        state_not_walls = self.policy_agent.compute_walls_from_transition_matrix(
            previous_action, self.env.p[:, self.env.state_from_coordinates(state[0], state[1]), :])

        # check if state is already in the states space and get their index
        state_not_walls_index = [key for key, value in states_space_policy.items()
            if np.equal(value, state_not_walls).all()]

        if len(state_not_walls_index) < 1: # if not, add it
            # add new states to states space
            states_space_policy[len(states_space_policy)] = state_not_walls
            
            # update policy: epsilon greedy
            if np.sum(state_not_walls) > 1:
                state_policy = state_not_walls * self.epsilon / (np.sum(state_not_walls) - 1)
                state_policy[np.where(state_not_walls == 1)[0][0]] = 1 - self.epsilon # first action on agent's left
            else:
                state_policy = state_not_walls.astype(float)
            policy.append(policy)
            
            # update advantages vector
            advantages_policy = np.concatenate((advantages_policy, np.zeros(N_ACTIONS)))

        # policy improvement using actor-critic
        action_agent_pov = ACTIONS_MAP[previous_action][action]
        gradient = softmax_gradient(policy=policy[state_not_walls_index[0]], 
            action=action_agent_pov)
        gradient = np.multiply(gradient, state_not_walls)
        # stochastic gradient ascent
        policy[state_not_walls_index[0]] += self.lr * gradient * advantages_model[
            states_space_model[state]][WALLS_MAP[previous_action]]
        
        # update advantages vector
        advantages_policy[state_not_walls_index[0]] += advantages_model[
            states_space_model[state]][WALLS_MAP[previous_action]]

        return policy, states_space_policy, advantages_policy