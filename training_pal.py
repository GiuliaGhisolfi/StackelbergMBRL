from src.algorithms.PAL import PAL

def main(learning_rate, n_environments, max_iterations_per_environment, n_episodes_per_iteration, max_epochs_per_episode, 
    maze_width, maze_height, alpha, gamma):
    pal = PAL(
        learning_rate=learning_rate, 
        n_environments=n_environments,
        max_iterations_per_environment=max_iterations_per_environment,
        n_episodes_per_iteration=n_episodes_per_iteration, 
        max_epochs_per_episode=max_epochs_per_episode, 
        maze_width=maze_width, 
        maze_height=maze_height, 
        alpha = alpha,
        gamma=gamma)
    
    pal.train()


if __name__ == '__main__':
    # maze parameters
    maze_width = 101
    maze_height = 51

    # agent parameters
    gamma = 0.9 # discount factor to compute expected cumulative reward
    alpha = 0.01 # learning rate for policy improvment

    # training parameters
    n_environments = 2 # number of different environments to train on
    max_iterations_per_environment = 3
    n_episodes_per_iteration = 4 # number of episodes to run for each epoch
    max_epochs_per_episode = 600
    learning_rate = 0.01


    main(
        learning_rate=learning_rate, 
        n_environments=n_environments,
        max_iterations_per_environment=max_iterations_per_environment,
        n_episodes_per_iteration=n_episodes_per_iteration, 
        max_epochs_per_episode=max_epochs_per_episode, 
        maze_width=maze_width, 
        maze_height=maze_height, 
        alpha=alpha,
        gamma=gamma
    )
