import numpy as np
import nashpy as nash
from keras.losses import KLDivergence

def compute_model_loss(transition_probability_model, transition_probability_data):
    # compute loss between transition_probability_model and transition_probability_data
    kl_divergence = compute_kl_divergence(
        y_true=transition_probability_model, 
        y_pred=transition_probability_data
        )
    loss = kl_divergence.sum() # sum over all actions
    return loss

def compute_kl_divergence(y_true, y_pred):
    # compute KL divergence between y_true and y_pred
    kl_divergence = KLDivergence()
    # TODO: controll + raise error or something if kl is negative
    return kl_divergence(y_true, y_pred).numpy()

def stackelberg_nash_equilibrium(leader_payoffs, follower_payoffs):
    # initialize the game
    game = nash.Game(leader_payoffs, follower_payoffs)

    # Find the Stackelberg equilibrium using the support enumeration algorithm
    stackelberg_equilibria = list(game.support_enumeration())

    # Extract the probabilities of the leader and the strategy of the follower
    leader_probabilities, follower_strategy = stackelberg_equilibria[0]

    return follower_strategy