import numpy as np
from src.agents.model_agent import ModelAgent
from src.agents.policy_agent import PolicyAgent
from src.environment.environment import Environment
from src.algorithms.utils import softmax_gradient, softmax, stackelberg_nash_equilibrium, \
    compute_model_loss, compute_actor_critic_objective, save_metrics, save_policy, save_parameters

ACTION_LIST = [0, 1, 2, 3] # [up, down, left, right]

class MAL():
    # MAL: Model As Leader Algorithm

    def __init__(self, learning_rate, n_environments, max_iterations_per_environment, n_episodes_per_iteration, 
        max_epochs_per_episode, maze_width, maze_height, beta, gamma, epsilon, temperature):

        self.n_environments = n_environments
        self.max_iterations_per_environment = max_iterations_per_environment
        self.n_episodes_per_iteration = n_episodes_per_iteration
        self.max_epochs_per_episode = max_epochs_per_episode
        self.lr = learning_rate # learning rate for policy improvement
        self.beta = beta # alpha, learning rate for value function update
        self.gamma = gamma # discount factor to compute expected cumulative reward
        self.epsilon = epsilon # epsilon greedy parameter
        self.temperature = temperature # temperature for softmax gradient

        parameters_dict = {
            'n_environments': self.n_environments,
            'max_iterations_per_environment': self.max_iterations_per_environment,
            'n_episodes_per_iteration': self.n_episodes_per_iteration,
            'max_epochs_per_episode': self.max_epochs_per_episode,
            'learning_rate': self.lr,
            'alpha': self.beta,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'temperature': self.temperature
            }
        save_parameters(parameters_dict, algorithm='PAL') # save parameters in json file

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
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],
            epsilon=epsilon
            )
        print('Agents initialized')

        # transition probability matrix
        self.p = self.model_agent.transition_distribuition[
            self.model_agent.agent_state]
    
    def train(self):
        # train loop over n_environments environments
        print('Training started')
        for i in range(self.n_environments):
            print(f'\nTraining on environment {i+1}/{self.n_environments}')
            # train loop for the environment
            self.__train_loop_for_the_environment(environment_number=i+1)

            # checkpoint: save policy, policy states space and metrics in json file
            save_policy(policy=self.policy_agent.policy, states_space=self.policy_agent.states_space, 
                algorithm='MAL', environment_number=i)

            if i < self.n_environments - 1:
                # reset and initialize new environment and model agent
                self.env.reset_environment()
                self.model_agent.reset_agent(
                    initial_state_coord=self.env.initial_state_coord, 
                    transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])

                # reset transition probability matrix p
                self.p = self.model_agent.transition_distribuition[self.model_agent.agent_state]

                # reset policy agent at initial state
                self.policy_agent.reset_at_initial_state(
                    transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])
    
    def reset_at_initial_state(self):
        # reset agent's position in the environment
        self.env.reset()

        # reset agent state at initial state
        self.model_agent.agent_state = 0
        self.policy_agent.reset_at_initial_state(
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])
    
    def __train_loop_for_the_environment(self, environment_number):
        # train loop for the environment
        for i in range(self.max_iterations_per_environment):
            print(f'\nIteration {i+1}/{self.max_iterations_per_environment}')
            # train loop for the iteration
            self.__train_loop_for_the_iteration(environment_number=environment_number, iteration_number=i+1)

            # TODO: update

    def __train_loop_for_the_iteration(self, environment_number, iteration_number):
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
 
########################################################################################################
def optimize_policy(policy, model, env):
    # optimize policy massimizing reward given the model
    #policy = argmax
    return policy

def improve_model(model, policy, data_buffer, beta):
    # improve model using Gradient Descent
    return model