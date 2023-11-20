import tensorflow as tf

from src.agent import Agent
from src.algorithms.utils import executing_policy
from keras.losses import KLDivergence


class PAL(Agent):
    # PAL: Policy As Leader Algorithm

    def __init__(self, gamma, initial_state_coord, transition_matrix_initial_state, learning_rate, n_episodes_per_iteration):
        super().__init__(gamma, initial_state_coord, transition_matrix_initial_state)
        self.n_episodes_per_iteration = n_episodes_per_iteration
        self.lr = learning_rate # alpha

    def PAL(self, policy, model, env):
        data_buffer = [] # list of episodes, each episode is a list of tuples (state, action, reward, next_state)

        # collect data executing policy in the environment
        for _ in range(self.n_episodes_per_iteration):
            episode = executing_policy(policy, env)
            data_buffer.append(episode)
        
        # build policy-specific model
        model = optimize_model(model, data_buffer)

        # improve policy (TRPO)
        #policy = trpo(policy, model, self.lr) #TODO: cambiare, vuole solo env con lo standard di OpenAI come input
        trpo(env) #FIXME: non lo so ma dovrebbe essere una funzione che copia l'enviroment, poi deve ritornare la policy

        return policy, model

def optimize_model(model, data_buffer):
    # build model given data_buffer
    # minimizzo KL div tra modello precedente (matrice di transizione P) e 
    # approx del modello ricavata dai dati (matrice Q di approx di P dagli episodi in data_buffer)
    for episode in data_buffer:
        q = compute_transition_probability_from_episode(episode)
        loss = compute_model_loss(
            transition_probability_model=model.p, 
            transition_probability_data=q)

    return model

def compute_kl_divergence(y_true, y_pred):
    # compute KL divergence between y_true and y_pred
    kl_divergence = KLDivergence()
    return kl_divergence(y_true, y_pred).numpy()

def compute_model_loss(transition_probability_model, transition_probability_data):
    kl_divergence = compute_kl_divergence(
        y_true=transition_probability_model, 
        y_pred=transition_probability_data
        )
    loss = kl_divergence.sum() # sum over all actions
    return loss
