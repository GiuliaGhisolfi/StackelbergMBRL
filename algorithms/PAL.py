from utils import executing_policy

# PAL: Policy As Leader Algorithm
def PAL(self, policy, model, env, n_episodes_per_iteration, alpha):
    data_buffer = [] # list of tuples (state, action, reward, next_state)

    # collect data executing policy in the environment
    for iteration in range(n_episodes_per_iteration):
        episode = executing_policy(self, policy, env)
        data_buffer.append(episode)
    
    # build policy-specific model
    model = optimize_model(model, policy, data_buffer)

    # improve policy (TRPO)
    policy = TRPO(policy, model, alpha)

    return policy, model

def TRPO(policy, model, alpha):
    # improve policy using TRPO
    return policy

def optimize_model(model, policy, data_buffer):
    # build model given data_buffer
    return model