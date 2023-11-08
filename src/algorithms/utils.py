from environment import Environment

# Data collection
def executing_policy(policy, env):
    # execute policy in the environment to collect data
    current_state = env.agent_state
    episode = []
    while current_state != env.goal:
        action = policy(current_state)
        next_state = env.maze[current_state][action]
        reward = env.compute_reward(current_state, action)
        episode.append((current_state, action, reward, next_state))
        current_state = next_state
    return episode