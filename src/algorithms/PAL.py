import numpy as np
from keras.losses import KLDivergence
from src.agents.model_agent import ModelAgent
from src.agents.policy_agent import PolicyAgent
from src.environment.environment import Environment
from src.algorithms.utils import executing_policy

ACTION_LIST = [0, 1, 2, 3] # [up, down, left, right]

def compute_model_loss(transition_probability_model, transition_probability_data):
    kl_divergence = compute_kl_divergence(
        y_true=transition_probability_model, 
        y_pred=transition_probability_data
        )
    loss = kl_divergence.sum() # sum over all actions
    return loss

def compute_kl_divergence(y_true, y_pred):
    # compute KL divergence between y_true and y_pred
    kl_divergence = KLDivergence()
    return kl_divergence(y_true, y_pred).numpy()


class PAL():
    # PAL: Policy As Leader Algorithm

    def __init__(self, learning_rate, n_episodes_per_iteration, maze_width, maze_height, gamma):
        self.n_episodes_per_iteration = n_episodes_per_iteration
        self.lr = learning_rate # alpha

        # initalize environment
        self.env = Environment(
            maze_width=maze_width,
            maze_height=maze_height
            )

        # initialize model and policy agents
        self.model_agent = ModelAgent(
            gamma=gamma,
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],
            initial_state_coord=self.env.initial_state_coord,
            )
        self.policy_agent = PolicyAgent(
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],
            )
    
    def reset(self):
        # reset agent's position in the environment
        self.env.reset()

        # reset agent state
        self.model_agent.agent_state = 0 # initial state
        # TODO
    
    def PAL(self):
        data_buffer = [] # list of episodes, each episode is a list of tuples (state, action, reward, next_state)

        # collect data executing policy in the environment
        for _ in range(self.n_episodes_per_iteration):
            episode = executing_policy(self.policy_agent, self.env)
            data_buffer.append(episode)
            self.reset() # reset environment and agent state
        
        # build policy-specific model
        self.model_agent.transition_distribuition = self.__optimize_model(
            self.model_agent.transition_distribuition, data_buffer)
        
        # improve policy
        #TODO

    def __optimize_model(self, transition_distribuition, data_buffer):
        # build model given data_buffer
        # minimizzo KL div tra modello precedente (matrice di transizione P) e 
        # approx del modello ricavata dai dati (matrice Q di approx di P dagli episodi in data_buffer)
        loss = np.inf
        self.state_space = self.model_agent.state_space

        for episode in data_buffer:
            q = self.__compute_transition_probability_from_episode(episode)
            p = self.__comunpute_transition_probability_from_model(self.model_agent.transition_distribuition)
            local_loss = compute_model_loss(
                transition_probability_model=p,
                transition_probability_data=q)
            if local_loss < loss:
                loss = local_loss
                p = q

        return p
    
    def __compute_transition_probability_from_episode(self, episode):
        # compute transition probability from episode
        # episode = [(state, action, reward, next_state), ...]
        states_space_dim = len(self.state_space)
        q = np.zeros(states_space_dim, len(ACTION_LIST))
        for state, action, _, _ in episode:
            if state not in self.state_space.keys():
                states_space_dim += 1
                self.state_space[state] = states_space_dim-1
            q[self.state_space[state], action] += 1
            #TODO: update netx_sate_function and reward_function ?
        q /= np.sum(q, axis=1)
        return q
    
    def __comunpute_transition_probability_from_model(self, transition_distribuition):
        # compute transition probability from model
        # transition_distribuition = {state: probability distribution over actions}
        p = np.zeros(len(self.states_space_dim), len(ACTION_LIST))
        for state, action in transition_distribuition.keys():
            p[state, action] = transition_distribuition[state, action]
        return p

def compute_cost_function(policy_agent, policy_data):
    # cost function to maximaize: expected cumulative reward executing a policy in the environment
    # transition matrix is the one from the model
    #TODO: to improve policy
    pass
