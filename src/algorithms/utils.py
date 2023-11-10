import numpy as np
from environment import Environment

ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]

# Data collection
def executing_policy(policy, env):
    # execute policy in the environment to collect data
    current_state = env.state
    episode = []

    while current_state != env.terminal_state:
        # select action from policy
        action = select_action(policy, env, current_state)

        # compute step in environment
        next_state, reward, _, _, _ = env.step(action)

        # save step in episode
        episode.append((current_state, action, reward, next_state))

        # update current state
        current_state = next_state
        
    return episode

def select_action(policy, env, current_state):
    if current_state in policy:
        current_state_coord = env.coordinates_from_state(current_state)
        return np.random.choice(ACTIONS_LIST, policy[current_state_coord])
    else:
        # possible actions from current state
        action_list = np.where(env.p[:, current_state, :] != 0)[1]
        return np.random.choice(action_list)
    