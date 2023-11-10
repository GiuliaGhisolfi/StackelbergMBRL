from utils import executing_policy
from spinningup.spinup.algos.tf1.trpo import trpo

# PAL: Policy As Leader Algorithm
def PAL(policy, model, env, n_episodes_per_iteration, alpha):
    data_buffer = [] # list of tuples (state, action, reward, next_state)

    # collect data executing policy in the environment
    for iteration in range(n_episodes_per_iteration):
        episode = executing_policy(self, policy, env)
        data_buffer.append(episode)
    
    # build policy-specific model
    model = optimize_model(model, policy, data_buffer)

    # improve policy (TRPO)
    policy = trpo(policy, model, alpha) #TODO: cambiare, vuole solo env con lo standard di OpenAI come input

    return policy, model

def optimize_model(model, policy, data_buffer):
    # build model given data_buffer
    return model