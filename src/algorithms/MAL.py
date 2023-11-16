from src.agent import Agent
from src.algorithms.utils import executing_policy


class MAL(Agent):
    def __init__(self, gamma, initial_state_coord, transition_matrix_initial_state):
        super().__init__(gamma, initial_state_coord, transition_matrix_initial_state)

    # MAL: Model As Leader Algorithm
    def MAL(policy, model, env, n_episodes_per_iteration, alpha):
        data_buffer = []
        # optimize policy massimizing reward given the model
        policy = optimize_policy(policy, model)

        # collect data executing optimized policy in the environment
        for _ in range(n_episodes_per_iteration):
            episode = executing_policy(policy, env)
            data_buffer.append(episode)
        
        # improve model (Gradient Descent o altro)
        model = GD(model, policy, data_buffer, alpha)

        return policy, model

    def optimize_policy(policy, model, env):
        # optimize policy massimizing reward given the model
        return policy

    def GD(model, policy, data_buffer, alpha):
        # improve model using Gradient Descent
        return model