from src.algorithms.PAL import PAL

def main(alpha, n_environments, max_iterations_per_environment, n_episodes_per_iteration, max_epochs_per_episode, 
    maze_width, maze_height, gamma):
    pal = PAL(
        learning_rate=alpha, 
        n_environments=n_environments,
        max_iterations_per_environment=max_iterations_per_environment,
        n_episodes_per_iteration=n_episodes_per_iteration, 
        max_epochs_per_episode=max_epochs_per_episode, 
        maze_width=maze_width, 
        maze_height=maze_height, 
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
    n_environments = 1 # number of different environments to train on
    max_iterations_per_environment = 2
    n_episodes_per_iteration = 20 # number of episodes to run for each epoch
    max_epochs_per_episode = 6000


    main(
        alpha=alpha, 
        n_environments=n_environments,
        max_iterations_per_environment=max_iterations_per_environment,
        n_episodes_per_iteration=n_episodes_per_iteration, 
        max_epochs_per_episode=max_epochs_per_episode, 
        maze_width=maze_width, 
        maze_height=maze_height, 
        gamma=gamma
    )
