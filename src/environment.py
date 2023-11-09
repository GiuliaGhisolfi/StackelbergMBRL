import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import spy

from matrix_mdp.envs import MatrixMDPEnv
from src.maze import Maze
from scipy.spatial.distance import cityblock

THRESHOLD_INITIAL_TERMINAL_DISTANCE_PERCENTAGE = 0.5 # TODO: da cambiare o mettere come parametro

class Environment(MatrixMDPEnv):
    
    def __init__(self, maze_width, maze_height):
        self.maze_width = maze_width
        self.maze_height =  maze_height

        # initializa maze
        maze = Maze(size_x=int(self.maze_width/2), size_y=int(self.maze_height/2))
        self.maze = maze.blocks
        self.flatten_maze = maze.blocks.flatten().reshape(1, -1)

        # initialize initial state
        print('Compute initial state')
        self.compute_initial_states()

        # compute prior and transitional distribuitions
        print('Compute prior and transitional distribuitions')
        self.compute_prior_distribuitions()
        self.compute_transition_distribuition()

        # initialize terminal state translation probability
        print('Compute terminal state')
        self.compute_terminal_states()

        # compute reward function
        print('Compute reward function')
        self.compute_reward_function()

        super.__init__(p_0=self.p_0, p=self.p, r=self.r, render_mode='human')
        print('Environment created')
    
    def compute_initial_states(self):
        self.n_states =  np.sum(self.flatten_maze == 0) # states cardinality: walkable cells
        self.n_actions = 4 # actions cardinality: up, down, left, right

        # sampling initial state from a uniform distribution over all possible states
        self.initial_state = np.random.choice(np.arange(self.n_states))
 
    def compute_prior_distribuitions(self):
        # compute prior distribuition w.r.t. initial state
        self.p_0 = np.zeros(self.n_states)
        self.p_0[self.initial_state] = 1
    
    def compute_transition_distribuition(self):
        #TODO: si può scrivere meglio?
        # compute transition probability matrix given to the environment P(S'|S,A)
        self.p = np.ones((self.n_states, self.n_states, self.n_actions)) / self.n_states # init
        actions_list = [0, 1, 2, 3] # up, down, left, right

        for state in range(self.n_states):
            x, y = self.coordinates_from_state(state)

            # compute neighbors: state agent can go from current state in (x,y)
            neighbors_coord = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)] # up, down, left, right
            neighbors = [self.maze[coord] for coord in neighbors_coord]
            den = np.sum(neighbors)

            # compute probability to transited to neighbors from current state in (x,y)
            for neighbor, (y_neighbor, x_neighbor), action in zip(neighbors, neighbors_coord, actions_list):
                if neighbor == 0:
                    neighbor_state = self.state_from_coordinates(x_neighbor, y_neighbor)
                    self.p[neighbor_state, state, action] = 1/den
        
    def compute_terminal_states(self):
        initial_state_coord = self.coordinates_from_state(self.initial_state)
        manhattan_distance = -1 # init
        threshold_initial_terminal_distance = THRESHOLD_INITIAL_TERMINAL_DISTANCE_PERCENTAGE * self.n_states

        while manhattan_distance < threshold_initial_terminal_distance:
            # sampling terminal state from a uniform distribution over all possible states
            self.terminal_state = np.random.choice(np.arange(self.n_states)) 
            #TODO: si può fare meglio, magari cambiando il sampling dello stato iniziale e cercando di non averlo nel centro

            # convert state in gridword coordinates
            terminal_state_coord = self.coordinates_from_state(self.terminal_state)

            # calculate Manhattan distance between inital and terminal states
            manhattan_distance = cityblock(initial_state_coord, terminal_state_coord)

        # set the translation probability from the terminal state to zero
        self.p[:, self.terminal_state, :] = np.zeros((self.n_states, self.n_actions))
    
    def compute_reward_function(self):
        # compute reward function penalizing the agent for each step taken
        self.r = -np.ones((self.n_states, self.n_states, self.n_actions))

        # penalize state with only one neighbor
        for state in range(self.n_states):
            if np.sum(np.nonzero(self.p[:, state, :])) == 1:
                self.r[:, state, :] = -2

        # set the reward for the terminal state to zero
        self.r[:, self.terminal_state, :] = np.zeros((self.n_states, self.n_actions))

    def coordinates_from_state(self, state):
        # count nnz elements in flatten maze until state-th zero element
        temporary = np.nonzero(self.flatten_maze == 0)[1][state]

        # convert state from temporary position in gridword coordinates
        x = temporary % self.maze_width
        y = int(temporary / self.maze_width)

        return x, y

    def state_from_coordinates(self, x, y):
        # convert gridword coordinates in state
        temporary = np.zeros(self.flatten_maze.shape)
        temporary[0, x + self.maze_width * y] = 1
        temporary = temporary[np.where(self.flatten_maze == 0)]

        return np.where(temporary==1)[0][0]
    
    def display_maze(self):
        plt.figure(figsize=(10, 10))
        spy(self.maze)
        plt.xticks(np.arange(0, self.maze.shape[1], 2))
        plt.yticks(np.arange(0, self.maze.shape[0], 2))
        plt.grid(axis='both', which='both')