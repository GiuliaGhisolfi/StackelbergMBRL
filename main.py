from src.mbrl_maze_solver import MBRLMazeSolver

def main(maze_width, maze_height, max_epochs, algorithm, n_episodes_per_iteration, 
        gamma, alpha, beta, render, render_wait):
    mbrl_maze_solver = MBRLMazeSolver( 
        maze_width=maze_width, 
        maze_height=maze_height,
        max_epochs=max_epochs, 
        algorithm=algorithm,
        n_episodes_per_iteration=n_episodes_per_iteration, 
        gamma=gamma, 
        alpha=alpha, 
        beta=beta,
        render=render,
        render_wait=render_wait
    )

    mbrl_maze_solver.run()


if __name__ == '__main__':
    # maze parameters
    maze_width = 101
    maze_height = 51
    render = True
    render_wait = 0 # time to wait between frames in ms

    # agent parameters
    gamma = 0.9 # discount factor to compute expected cumulative reward
    algorithm = 'PAL' # 'PAL' or 'MAL' or 'baseline'

    # PAL agent parameters
    alpha = 0.01 # learning rate for policy improvment

    # MAL agent parameters
    beta = 0.01 # learning rate for model update

    # training parameters
    max_epochs = 6000
    n_episodes_per_iteration = 20 # number of episodes to run for each epoch


    main(maze_width=maze_width, 
        maze_height=maze_height,
        max_epochs=max_epochs, 
        algorithm=algorithm,
        n_episodes_per_iteration=n_episodes_per_iteration, 
        gamma=gamma, 
        alpha=alpha, 
        beta=beta,
        render=render,
        render_wait=render_wait
    )
