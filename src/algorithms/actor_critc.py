import numpy as np
#TODO: cancellare questo file

def update_weights(self, states, actions, rewards, alpha=0.1, gamma=0.9): # compute value function
    # Aggiornamento dei pesi della policy e del critic utilizzando l'algoritmo Actor-Critic
    for i in range(len(states)):
        state = states[i]
        action = actions[i]
        reward = rewards[i]

        # Calcolo dell'avantage come differenza tra la ricompensa e la value function
        advantage = reward + gamma * self.value_function(states[i + 1]) - self.value_function(state)

        # Aggiornamento dei pesi della policy utilizzando il gradiente softmax ponderato dall'avantage
        action_probs = self.policy(state)
        self.actor_weights[state] += alpha * advantage * (self.one_hot(action, len(action_probs)) - action_probs)

        # Aggiornamento dei pesi del critic utilizzando il TD error
        td_error = reward + gamma * self.value_function(states[i + 1]) - self.value_function(state)
        self.critic_weights[state] += alpha * td_error # value function


############ My code #############
def advantage_actor_critic(states, actions, rewards, value_function, alpha, gamma, num_episodes, temperature):
    # Initialize policy weights randomly uniformly
    policy_weights = np.random.rand(len(states))

    for episode in range(num_episodes):
        advantages = calculate_advantages(states, rewards, value_function, gamma)

        for i in range(len(states)):
            current_state = states[i]
            current_action = actions[i]

            # compute softmax gradient
            softmax_grad = softmax_gradient(policy_weights, states, temperature, action=current_action)

            # weight update
            policy_weights += alpha * advantages[i] * softmax_grad

    return policy_weights

def calculate_advantages(states, rewards, value_function, gamma):
    advantages = np.zeros(len(states)) #TD error
    for i in range(len(states)):
        current_state = states[i]
        next_state = states[i + 1] if i + 1 < len(states) else None
        reward = rewards[i]

        # compute advantage as difference between reward and value function estimate
        if next_state is not None:
            next_value = value_function[next_state]
        else:
            next_value = 0  # last visited state
        advantages[i] = reward + gamma * next_value - value_function[current_state]

    return advantages

def compute_actor_critic_objective(policy_probs, advantages):
    # compute weighted log-likelihoods from advantages
    log_likelihoods = np.log(policy_probs)
    weighted_log_likelihoods = log_likelihoods * advantages

    # objective function is sum of weighted log-likelihoods
    objective = np.sum(weighted_log_likelihoods)

    return objective # cost
