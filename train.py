from src.algorithms.MAL import MAL
from src.algorithms.PAL import PAL

def main(algorithm, learning_rate, n_environments, max_iterations_per_environment, n_episodes_per_iteration, 
    max_epochs_per_episode, maze_width, maze_height, alpha, gamma, epsilon, verbose):
    if algorithm == 'MAL':
        alg = MAL(
            learning_rate=learning_rate, 
            n_environments=n_environments,
            max_iterations_per_environment=max_iterations_per_environment,
            n_episodes_per_iteration=n_episodes_per_iteration, 
            max_epochs_per_episode=max_epochs_per_episode, 
            maze_width=maze_width, 
            maze_height=maze_height, 
            alpha = alpha,
            gamma=gamma,
            epsilon=epsilon,
            verbose=verbose
            )
    elif algorithm == 'PAL':
        alg = PAL(
            learning_rate=learning_rate, 
            n_environments=n_environments,
            max_iterations_per_environment=max_iterations_per_environment,
            n_episodes_per_iteration=n_episodes_per_iteration, 
            max_epochs_per_episode=max_epochs_per_episode, 
            maze_width=maze_width, 
            maze_height=maze_height, 
            alpha = alpha,
            gamma=gamma,
            epsilon=epsilon,
            verbose=verbose
            )
    
    alg.train()


if __name__ == '__main__':
    algorithm = 'PAL' # 'PAL' or 'MAL'

    # maze parameters
    maze_width = 101
    maze_height = 51

    # agent parameters
    gamma = 0.995 # discount factor to compute expected cumulative reward
    alpha = 0.01 # learning rate for policy improvment

    # training parameters
    n_environments = 5 # number of different environments to train on
    max_iterations_per_environment = 10
    n_episodes_per_iteration = 3 # number of episodes to run for each epoch
    max_epochs_per_episode = 1000
    learning_rate = 0.1
    epsilon = 0.1 # epsilon for epsilon-greedy policy


    main(
        algorithm=algorithm,
        learning_rate=learning_rate, 
        n_environments=n_environments,
        max_iterations_per_environment=max_iterations_per_environment,
        n_episodes_per_iteration=n_episodes_per_iteration, 
        max_epochs_per_episode=max_epochs_per_episode, 
        maze_width=maze_width, 
        maze_height=maze_height, 
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        verbose=True
    )
