import numpy as np


# Data collection
def executing_policy(agent, env):
    """
    Execute policy in the environment to collect data

    Args:
        agent (Agent)
        env (Environment)

    Returns:
        list os steps: [(state, action, reward, next_state), ...]
    """

    current_state = env.state
    episode = [] # init

    while current_state != env.terminal_state:
        # select action from policy
        action = select_action(agent.policy_agent, env, current_state)

        # compute step in environment
        next_state, reward, _, _, _ = env.step(action)

        # save step in episode
        episode.append((current_state, action, reward, next_state))

        # update current state
        current_state = next_state
        
    return episode

def select_action(policy_agent, env, current_state):
    if current_state in policy_agent.policy.keys():
        current_state_coord = env.coordinates_from_state(current_state)
        return policy_agent.take_action(current_state_coord)
    else:
        # select possible actions from current state with uniform distribution
        action_list = np.where(env.p[:, current_state, :] != 0)[1]
        return np.random.choice(action_list)

    