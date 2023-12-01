from src.algorithms.PAL import PAL

def main(learning_rate, n_environments, max_iterations_per_environment, n_episodes_per_iteration, max_epochs_per_episode, 
    maze_width, maze_height, alpha, gamma, epsilon, temperature):
    pal = PAL(
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
        temperature=temperature
        )
    
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
    max_iterations_per_environment = 2
    n_episodes_per_iteration = 2 # number of episodes to run for each epoch
    max_epochs_per_episode = 600
    learning_rate = 0.01
    epsilon = 0.1 # epsilon for epsilon-greedy policy
    temperature = 1 # temperature for softmax policy


    main(
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
        temperature=temperature
    )
