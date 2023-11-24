import nashpy as nash

def stackelberg_nash_equilibrium(leader_payoffs, follower_payoffs):
    # initialize the game
    game = nash.Game(leader_payoffs, follower_payoffs)

    # Find the Stackelberg equilibrium using the support enumeration algorithm
    stackelberg_equilibria = list(game.support_enumeration())

    # Extract the probabilities of the leader and the strategy of the follower
    leader_probabilities, follower_strategy = stackelberg_equilibria[0]

    return follower_strategy