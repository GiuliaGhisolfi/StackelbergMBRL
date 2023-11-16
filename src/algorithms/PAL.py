from src.agent import Agent
from src.algorithms.utils import executing_policy
#from spinningup.spinup.algos.tf1.trpo import trpo


class PAL(Agent):
    def __init__(self, gamma, initial_state_coord, transition_matrix_initial_state):
        super().__init__(gamma, initial_state_coord, transition_matrix_initial_state)

    # PAL: Policy As Leader Algorithm
    def PAL(policy, model, env, n_episodes_per_iteration, alpha):
        data_buffer = [] # list of tuples (state, action, reward, next_state)

        # collect data executing policy in the environment
        for _ in range(n_episodes_per_iteration):
            episode = executing_policy(policy, env)
            data_buffer.append(episode)
        
        # build policy-specific model
        model = optimize_model(model, policy, data_buffer)

        # improve policy (TRPO)
        #policy = trpo(policy, model, alpha) #TODO: cambiare, vuole solo env con lo standard di OpenAI come input
        trpo(env) #FIXME: non lo so ma dovrebbe essere una funzione che copia l'enviroment, poi deve ritornare la policy

        return policy, model

    def optimize_model(model, data_buffer):
        # build model given data_buffer
        # minimizzo KL div tra modello precedente e approx del modello ricavata dai dati
        return model