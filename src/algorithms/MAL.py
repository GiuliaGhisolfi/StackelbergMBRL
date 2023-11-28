from src.agent import Agent
from src.algorithms.utils import executing_policy


class MAL(Agent): #TODO: non deve estendere Agent class
    # MAL: Model As Leader Algorithm

    def __init__(self, gamma, initial_state_coord, transition_matrix_initial_state, learning_rate, n_episodes_per_iteration):
        super().__init__(gamma, initial_state_coord, transition_matrix_initial_state)
        self.n_episodes_per_iteration = n_episodes_per_iteration
        self.lr = learning_rate # beta

    def MAL(self, policy, model, env):
        data_buffer = [] # list of episodes, each episode is a list of tuples (state, action, reward, next_state)

        # optimize policy massimizing reward given the model
        self.policy = optimize_policy(self.policy, self.model)

        # collect data executing optimized policy in the environment
        for _ in range(self.n_episodes_per_iteration):
            episode = executing_policy(policy, env)
            data_buffer.append(episode)
        
        # improve model (Gradient Descent o altro)
        self.model = improve_model(self.model, self.policy, data_buffer, self.lr)

def optimize_policy(policy, model, env):
    # optimize policy massimizing reward given the model
    #policy = argmax
    return policy

def improve_model(model, policy, data_buffer, beta):
    # improve model using Gradient Descent
    return model