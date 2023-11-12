import numpy as np

from src.environment import Environment
from src.agent import Agent

def main(maze_width, maze_height, gamma):
    env = Environment(
            maze_width=maze_width, 
            maze_height=maze_height
            )
    agent = Agent(
            initial_state_coord=env.initial_state_coord, 
            transition_matrix_initial_state=env.p[:, env.initial_state, :],
            gamma=gamma, 
            actions_list_initial_state = np.where(env.p[:, env.initial_state, :] != 0)[1]
            )
    
    env.render()
    agent.render(env.window, env.block_pixel_size)


if __name__ == '__main__':
    # maze parameters
    maze_width = 61
    maze_height = 31

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
