import numpy as np
import nashpy as nash
from keras.losses import KLDivergence

def softmax_gradient(policy, action, temperature):
    # compute softmax probabilities
    probabilities = softmax(policy / temperature)

    # compute softmax gradient
    gradient = np.zeros_like(probabilities)
    gradient[action] += 1 - probabilities[action]
    for i in range(len(policy)):
        if i != action: gradient[i] -= probabilities[i]

    return gradient

def softmax(x):
    if len(x.shape) > 1:
        # matrix
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
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

def stackelberg_nash_equilibrium(leader_payoffs, follower_payoffs):
    # initialize the game
    game = nash.Game([leader_payoffs], [follower_payoffs])

    # Find the Stackelberg equilibrium using the support enumeration algorithm
    stackelberg_equilibria = list(game.support_enumeration())

    # Extract the probabilities of the leader and the strategy of the follower
    if len(stackelberg_equilibria) > 0:
        leader_probabilities, follower_strategy = stackelberg_equilibria[0]
    else:
        leader_probabilities, follower_strategy = -1, -1 # no equilibria found

    return np.array(follower_strategy)