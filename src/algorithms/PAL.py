from src.stackelberg_agent import StackelbergAgent
from src.algorithms.utils import executing_policy
from spinup.algos.tf1.trpo import trpo


class PAL(StackelbergAgent):
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
        model = optimize_model(model, policy, data_buffer)

        # improve policy (TRPO)
        #policy = trpo(policy, model, self.lr) #TODO: cambiare, vuole solo env con lo standard di OpenAI come input
        trpo(env) #FIXME: non lo so ma dovrebbe essere una funzione che copia l'enviroment, poi deve ritornare la policy

        return policy, model

def optimize_model(model, data_buffer):
    # build model given data_buffer
    # minimizzo KL div tra modello precedente e approx del modello ricavata dai dati
    return model