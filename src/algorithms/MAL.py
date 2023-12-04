import numpy as np
from src.algorithms.maze_solver_algorithm import MazeSolverAlgorithm
from src.algorithms.utils import save_metrics, save_policy, check_stackelberg_nash_equilibrium

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
            policy = self.optimize_policy(quality_function=self.model_agent.quality_function)
            policy_cost_function_list = [self.compute_policy_cost_function(policy=policy,
                quality_function=self.model_agent.quality_function)]
            optimal_policy = 0
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

            if check_stackelberg_nash_equilibrium(leader_payoffs=model_loss_list, 
                follower_payoffs=policy_cost_function_list, optimal_leader=optimal_model,
                optimal_follower=optimal_policy):
                print('Stackelberg-Nash equilibrium reached') # FIXME: payoff della stessa lunghezza
                #TODO: save metrics
                break
            else:
                print('Stackelberg-Nash equilibrium not reached')