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

def update_policy_fixed_steps(current_policy_td, transition_matrix, reward_matrix, 
        terminal_state, n_steps=1000, gamma=0.3, alpha=1): 
    #TODO: sistemare + commentare TD lambda
    num_states, num_actions = current_policy_td.shape

    for _ in range(n_steps):
        state = np.random.randint(num_states)  # Start from a random state

        action = np.random.choice(num_actions, p=current_policy_td[state])
        next_state = np.random.choice(num_states, p=transition_matrix[state, action])
        reward = reward_matrix[state, action, next_state]

        # TD update rule
        td_error = reward + gamma * np.max(current_policy_td[next_state]) - current_policy_td[state, action]
        current_policy_td[state, action] += alpha * td_error
        current_policy_td[state] /= np.sum(current_policy_td[state]) # normalize the policy

        state = next_state

        if state == terminal_state:
            break

    return current_policy_td

def stackelberg_nash_equilibrium(leader_payoffs, follower_payoffs):
    # initialize the game
    game = nash.Game(leader_payoffs, follower_payoffs)

    # Find the Stackelberg equilibrium using the support enumeration algorithm
    stackelberg_equilibria = list(game.support_enumeration())

    # Extract the probabilities of the leader and the strategy of the follower
    leader_probabilities, follower_strategy = stackelberg_equilibria[0]

    return follower_strategy