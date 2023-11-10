from environment import Environment
from agent import ModelAgent, PolicyAgent

def main(maze_width, maze_height):
    env = Environment(maze_width, maze_height)


if __name__ == '__main__':
    # maze parameters
    maze_width = 61
    maze_height = 31

    # policy agent parameters
    alpha = 0.01 # learning rate for policy improvment

    # model agent parameters
    beta = 0.01 # learning rate for model update

    # other parameters
    max_epochs = 1000
    k = 20 # number of episodes to run for each epoch

    data_buffer = [] # list of episodes, each episode is a list of tuples (state, action, reward, next_state)
