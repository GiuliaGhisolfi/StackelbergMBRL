from src.maze_solver import MazeSolver

def main(maze_width, maze_height, gamma):
    maze_solver = MazeSolver(maze_width, maze_height, gamma, algorithm='baseline', 
        max_epochs=6000, n_episodes_per_iteration=20)
    maze_solver.run()   

if __name__ == '__main__':
    # maze parameters
    maze_width = 101
    maze_height = 51

    # policy agent parameters
    alpha = 0.01 # learning rate for policy improvment
    gamma = 0.9 # discount factor to compute expected cumulative reward

    # model agent parameters
    beta = 0.01 # learning rate for model update

    # other parameters
    max_epochs = 1000
    k = 20 # number of episodes to run for each epoch

    data_buffer = [] # list of episodes, each episode is a list of tuples (state, action, reward, next_state)

    main(maze_width, maze_height, gamma)
