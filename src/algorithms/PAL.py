import numpy as np
import json
from src.agents.model_agent import ModelAgent
from src.agents.policy_agent import PolicyAgent
from src.environment.environment import Environment
from src.algorithms.utils import stackelberg_nash_equilibrium, compute_model_loss, compute_kl_divergence

ACTION_LIST = [0, 1, 2, 3] # [up, down, left, right]

class PAL():
    # PAL: Policy As Leader Algorithm

    def __init__(self, learning_rate, n_environments, max_iterations_per_environment, n_episodes_per_iteration, 
        max_epochs_per_episode, maze_width, maze_height, gamma):

        self.n_environments = n_environments
        self.max_iterations_per_environment = max_iterations_per_environment
        self.n_episodes_per_iteration = n_episodes_per_iteration
        self.max_epochs_per_episode = max_epochs_per_episode
        self.lr = learning_rate # alpha
        self.gamma = gamma # discount factor to compute expected cumulative reward

        # initalize environment
        self.env = Environment(
            maze_width=maze_width,
            maze_height=maze_height
            )

        # initialize model and policy agents
        self.model_agent = ModelAgent(
            gamma=gamma,
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],
            initial_state_coord=self.env.initial_state_coord
            )
        self.policy_agent = PolicyAgent(
            gamma=gamma,
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :]
            )
        
        # transition probability matrix
        self.p = self.model_agent.transition_distribuition[
            self.model_agent.agent_state]
    
    def train(self):
        # train loop over n_environments environments
        for _ in range(self.n_environments):
            # train loop for the environment
            self.__train_loop_for_the_environment()

            #TODO: save policy in json file every n iterations, idk how many

            # reset and initialize new environment and model agent
            self.env.reset_environment()
            self.model_agent.reset_agent(
                initial_state_coord=self.env.initial_state_coord, 
                transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])

            # reset transition probability matrix p
            self.p = self.model_agent.transition_distribuition[self.model_agent.agent_state]

            # reset policy agent at initial state
            self.policy_agent.reset_at_initial_state(transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])

        # save final policy and policy states space in json file #TODO: capire come salvare policy
        #with open('src/saved_policy/PAL_policy.json', 'w') as policy_file:
            #json.dump(self.policy_agent.policy, policy_file)
        #with open('src/saved_policy/PAL_states_space.json', 'w') as states_space_file:
            #json.dump(self.policy_agent.states_space, states_space_file)
    
    def reset_at_initial_state(self):
        # reset agent's position in the environment
        self.env.reset()

        # reset agent state at initial state
        self.model_agent.agent_state = 0
        self.policy_agent.reset_at_initial_state(
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])
 
    def __train_loop_for_the_environment(self):
        for _ in range(self.max_iterations_per_environment):
            data_buffer = [] # list of episodes, each episode is a list of tuples (state, action, reward, next_state)

            # collect data executing policy in the environment
            for _ in range(self.n_episodes_per_iteration):
                episode = self.executing_policy()
                data_buffer.append(episode)
                self.reset_at_initial_state() # reset environment and agent state
            
            # compute parameters to optimize model and improve policy
            p = np.array(self.model_agent.transition_distribuition) # transition probability matrix
            q_optimal, states_space_model, optimal_reward_function, optimal_next_state_function, \
                optimal_value_function, policy, states_space_policy = self.__optimize_model_improve_policy(data_buffer, p)
            
            # build policy-specific model
            self.model_agent.transition_distribuition = q_optimal
            self.model_agent.states_space = states_space_model
            self.model_agent.next_state_function = optimal_next_state_function
            self.model_agent.reward_function = optimal_reward_function
            self.model_agent.values_function = optimal_value_function
            
            # improve policy
            self.policy_agent.policy = policy 
            self.policy_agent.states_space = states_space_policy

            # stopping criteria: stop in nash equilibrium
            if self.__check_stackelberg_nash_equilibrium():
                break
        
    def executing_policy(self):
        """
        Execute policy in the environment to collect data

        Args:
            agent (PolicyAgent)
            env (Environment)

        Returns:
            list os steps: [(state, action, reward, next_state), ...]
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
                break
            
        return episode

    def __optimize_model_improve_policy(self, data_buffer, p):
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

        q_optimal = np.empty((0, len(ACTION_LIST))) # init data transition probability as an empty array
        optimal_states_space_model = dict() # init optimal states space
        optimal_value_function = dict() # init optimal value function

        # policy parameters
        policy = []
        states_space_policy = []
        cost_function = -np.inf # cost function to maximize
        self.cost_function = [] # list of cost function computed from each episode

        # loop over episodes in data buffer
        for i, episode in enumerate(data_buffer):
            # compute model parameters for model optimization
            q, temp_states_space_model, temp_reward_function, temp_next_state_function, temp_value_function = \
                self.__compute_transition_probability_from_episode(episode)
            q = self.__q_as_a_distribution(p, q)

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
            temp_policy, temp_states_space_policy = self.__improve_policy_from_episode(episode,
                value_function=temp_value_function, states_space_model=temp_states_space_model)
            local_cost_function = self.__compute_cost_function(temp_policy, temp_states_space_policy)

            if local_cost_function > cost_function:
                cost_function = local_cost_function
                policy = temp_policy
                states_space_policy = temp_states_space_policy
            self.cost_function.append(local_cost_function)

        return (q_optimal, optimal_states_space_model, optimal_reward_function, optimal_next_state_function, 
            optimal_value_function, policy, states_space_policy)
    
    ###### model optimization ######
    def __compute_transition_probability_from_episode(self, episode):
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

        for state, action, reward, next_state in episode:
            if state not in temp_states_space.keys():
                # add new visited state to states space
                temp_states_space[state] = states_model_space_dim 
                temp_value_function[states_model_space_dim] = 0 # init value function
                states_model_space_dim += 1

        # update transition probability matrix 
        q = np.zeros((states_model_space_dim, len(ACTION_LIST)))

        for state, action, reward, next_state in episode: 
            q[temp_states_space[state], action] += 1
            temp_reward_function[(temp_states_space[state], action)] = reward

            if next_state in temp_states_space.keys():
                # not add last state visited in the episode to next state function
                temp_next_state_function[(temp_states_space[state], action)] = temp_states_space[next_state]

        q /= np.sum(q, axis=1)[:, None]
        q[np.isnan(q)] = 0 # if 0/0, then 0

        # update value function using TD learning
        for state, action, reward, next_state in episode:
            if next_state in temp_states_space.keys():
                td_error = reward + (self.gamma  * temp_value_function[temp_states_space[next_state]] - 
                    temp_value_function[temp_states_space[state]])
            else:
                td_error = reward - temp_value_function[temp_states_space[state]]

            temp_value_function[temp_states_space[state]] += self.lr * td_error

        return q, temp_states_space, temp_reward_function, temp_next_state_function, temp_value_function

    def __q_as_a_distribution(self, p, q):
        """
        Compute q as a distribution over actions, 
        replacing rows of q that are all zeros with the corresponding rows of p

        Args:
            p (np.ndarray): transition probability matrix of the model
            q (np.ndarray): transition probability matrix of the data
        
        Returns:
            q (np.ndarray): transition probability matrix of the data as a distribution over actions
        """
        for i in range(len(q)):
            if np.all(q[i] == 0):
                q[i] = p[i]
        return q
    
    ###### policy improvement ######
    def __improve_policy_from_episode(self, episode, value_function, states_space_model):
        temp_policy = self.policy_agent.policy.copy()
        temp_states_space_policy = self.policy_agent.states_space.copy()
        previous_action = self.policy_agent.fittizial_first_action
            
        for state, action, reward, next_state in episode:
            # transition probability matrix of the environment
            temp_policy, temp_states_space_policy = self.__update_policy(state, temp_policy, 
                temp_states_space_policy, previous_action, value_function, states_space_model)
            
            previous_action = action
        
        return temp_policy, temp_states_space_policy
    
    def __update_policy(self, state, policy:list, states_space_policy:dict, previous_action:int,
        value_function:dict, states_space_model_model:dict):
        """
        Update given policy with new state if it is not already present in the given states space
        else update policy with new transition matrix and add new state to states space

        Args:
            policy (list): policy to update
            states_space_policy (dict): states space to update
            previos_action (int): action taken by the agent to arrive in the current state

        Returns:
            policy (list): updated policy
            states_space_policy (list): updated states space
        """
        state_not_walls = self.policy_agent.compute_walls_from_transition_matrix(
            previous_action, self.env.p[:, state, :])

        # check if state and next_state are already in the states space and get their index
        state_not_walls_index = [key for key, value in states_space_policy.items()
            if np.equal(value, state_not_walls).all()]

        if len(state_not_walls_index) < 1:
            #  add new states to states space
            states_space_policy[len(states_space_policy)] = state_not_walls
            policy.append(state_not_walls / np.sum(state_not_walls))

        # policy improvement
        gradient = 0 #FIXME: update using gradient descent (incremental value approximation)
        policy[state_not_walls_index[0]] += self.lr * gradient
        policy[state_not_walls_index[0]] -= min(policy[state_not_walls_index[0]]) # each action is in [0, 1]
        policy[state_not_walls_index[0]] /= np.sum(policy[state_not_walls_index[0]]) # sum to 1

        return policy, states_space_policy
    
    def __compute_cost_function(self, policy, states_space):
        """
        Compute cost function from policy and states space

        Args:
            policy (list): policy to evaluate
            states_space (dict): states space to evaluate

        Returns:
            cost_function (float): cost function computed from policy and states space
        """
        #TODO: sostituire con un metodo incrementale come gradient descent per value approximation
        cost_function = 0
        for state in states_space.keys(): #TODO: è sbagliato, solo per farlo funzionare
            cost_function += np.sum(policy[state])#*self.model_agent.reward_function[state])
        return cost_function

    ###### stopping criteria ######
    def __check_stackelberg_nash_equilibrium(self):
        # compute stackelberg nash equilibrium
        # return True if the policy is at nash equilibrium, False otherwise
        # verify if I'm in a equilbrium, knowing all possible transition distribuition from the model and the policy

        equilibria = np.where(stackelberg_nash_equilibrium(
            leader_payoffs=[-kl_div for kl_div in self.kl_divergence], 
            follower_payoffs=self.cost_function
            ) != 0)[0]

        return self.optimal_model in equilibria
