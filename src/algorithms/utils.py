import nashpy as nash
from src.agents.policy_agent import PolicyAgent
from src.environment.environment import Environment


# Data collection
def executing_policy(policy_agent: PolicyAgent, env: Environment, max_epochs_per_episode: int):
    """
    Execute policy in the environment to collect data

    Args:
        agent (PolicyAgent)
        env (Environment)

    Returns:
        list os steps: [(state, action, reward, next_state), ...]
    """

    current_state = env.state
    current_state_coord = env.coordinates_from_state(env.state)
    action = policy_agent.fittizial_first_action
    episode = [] # init

    for _ in range(max_epochs_per_episode):
        # select action from policy
        action = policy_agent.compute_next_action(action=action, 
                    transition_matrix=env.p[:, current_state, :])

        # compute step in environment
        next_state, reward, _, _, _ = env.step(action) # netx_state: numeric

        # save step in episode
        episode.append((current_state_coord, action, reward, env.coordinates_from_state(next_state)))

        # update current state
        current_state = next_state
        current_state_coord = env.coordinates_from_state(next_state)
        
        # check if terminal state is reached
        if current_state == env.terminal_state:
            break
        
    return episode
    
def stackelberg_nash_equilibrium(leader_payoffs, follower_payoffs):
    # initialize the game
    game = nash.Game(leader_payoffs, follower_payoffs)

    # Find the Stackelberg equilibrium using the support enumeration algorithm
    stackelberg_equilibria = list(game.support_enumeration())

    # Extract the probabilities of the leader and the strategy of the follower
    leader_probabilities, follower_strategy = stackelberg_equilibria[0]

    return follower_strategy