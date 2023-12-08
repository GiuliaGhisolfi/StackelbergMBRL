import json
import numpy as np
import nashpy as nash
from keras.losses import KLDivergence

###### GENERAL FUNCTIONS FOR PAL ######
def softmax_gradient(policy, action):
    # compute softmax probabilities
    probabilities = softmax(policy)

    # compute softmax gradient
    gradient = np.zeros_like(probabilities)
    gradient[action] += 1 - probabilities[action]
    for i in range(len(policy)):
        if i != action: gradient[i] -= probabilities[i]

    return gradient

def softmax(x):
    if len(np.array(x).shape) > 1:
        # matrix
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values.tolist() / np.sum(exp_values, axis=1, keepdims=True)
        probabilities = [state_probabilities for state_probabilities in probabilities] # convert to list of np.array
    else:
        # vector
        exp_values = np.exp(x - np.max(x))
        probabilities = exp_values / np.sum(exp_values)
    return probabilities

def compute_model_loss(transition_probability_model, transition_probability_data):
    # compute loss between transition_probability_model and transition_probability_data
    kl_div = kl_divergence(
        y_true=transition_probability_model, 
        y_pred=transition_probability_data[:transition_probability_model.shape[0], :]
        )
    loss = kl_div.sum() # sum over all actions
    if loss < 0:
        print(f'Negative kl divergence detected: kl divergence = {loss}')
        loss = 0.0 # avoid negative loss
    return loss

def kl_divergence(y_true, y_pred):
    # compute KL divergence between y_true and y_pred
    kl_divergence = KLDivergence()
    return kl_divergence(y_true, y_pred).numpy()

def compute_actor_critic_objective(policy_probs, advantages):
    # compute weighted log-likelihoods from advantages
    log_likelihoods = np.log(np.array(policy_probs) + 1e-10) # avoid log(0)

    weighted_log_likelihoods = np.zeros_like(log_likelihoods)
    for i, j in enumerate(advantages):  
        weighted_log_likelihoods[i] = log_likelihoods[i] * j

    # objective function is sum of weighted log-likelihoods
    objective = np.sum(weighted_log_likelihoods)

    return objective # cost function value

###### STACKELBERG NASH EQUILIBRIUM FUNCTIONS ######
def check_stackelberg_nash_equilibrium(leader_payoffs, follower_payoffs, equilibrium_find):
    """
    Check if the policy is at Stackelberg-Nash equilibrium, 
    knowing all possible transition distributions from the model and the policy

    Returns:
        bool: True if the policy is at Nash equilibrium, False otherwise
    """
    follower_eq = np.argmax(follower_payoffs) # best response
    leader_eq = np.argmax(leader_payoffs)

    if follower_eq != leader_eq:
        return False
    return (leader_eq, follower_eq) == equilibrium_find

###### SAVE AND LOAD FUNCTIONS ######
def save_parameters(parameters_dict:dict, algorithm:str):
    # save parameters in json file
    with open(f'training_parameters/{algorithm}_parameters.json', 'w') as parameters_file:
        json.dump(parameters_dict, parameters_file)
    
def save_policy(policy, states_space, algorithm, environment_number):
    # save final policy and policy states space in json file
    with open(f'training_parameters/{algorithm}/policy/policy_env_{environment_number}.json', 'w') as policy_file:
        json.dump([row.tolist() for row in policy], policy_file)

    with open(f'training_parameters/{algorithm}/policy/states_space_env_{environment_number}.json', 
        'w') as states_space_file:
        json.dump({str(key): value.tolist() for key, value in states_space.items()}, states_space_file)

def save_metrics(metrics_dict, environment_number, algorithm):
    with open(f'training_parameters/{algorithm}/metrics/metrics_env_{environment_number}.json', 
        'w') as metrics_file:
        json.dump(metrics_dict, metrics_file)