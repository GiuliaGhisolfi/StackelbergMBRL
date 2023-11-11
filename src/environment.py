import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import spy
import pygame

from matrix_mdp.envs import MatrixMDPEnv
from src.maze import Maze

class Environment(MatrixMDPEnv):
    
    def __init__(self, maze_width, maze_height):
        self.maze_width = maze_width
        self.maze_height =  maze_height
        self.window = None

        # initializa maze
        maze = Maze(size_x=int(self.maze_width/2), size_y=int(self.maze_height/2))
        self.maze = maze.blocks
        self.flatten_maze = maze.blocks.flatten().reshape(1, -1)

        # initialize initial and terminal state
        print('Compute initial and terminal state')
        self.compute_terminal_states()
        self.compute_initial_states()

        # compute prior and transitional distribuitions
        print('Compute prior and transitional distribuitions')
        self.compute_prior_distribuitions()
        self.compute_transition_distribuition()

        # compute reward function
        print('Compute reward function')
        self.compute_reward_function()

        super().__init__(p_0=self.p_0, p=self.p, r=self.r, render_mode='human')
        print('Environment created')
    
    def compute_initial_states(self):
        self.terminal_state_coord = self.coordinates_from_state(self.terminal_state)

        if self.terminal_state_coord[0] > self.maze_width/2: 
            x_inital_state = 1
            initial_state_up = True
        else: 
            x_inital_state = self.maze_width-2
            initial_state_up = False

        if self.terminal_state_coord[1] > self.maze_height/2: 
            y_initial_state = 1
            initial_state_left = True
        else: 
            y_initial_state = self.maze_height-2
            initial_state_left = False

        while self.maze[y_initial_state,x_inital_state]:
            if initial_state_up: 
                x_inital_state += 1
            else: x_inital_state -= 1
            if initial_state_left: 
                y_initial_state += 1
            else: y_initial_state -= 1

        self.initial_state = self.state_from_coordinates(x_inital_state, y_initial_state)
        self.initial_state_coord = (x_inital_state, y_initial_state)
    
    def compute_terminal_states(self):
        self.n_states =  np.sum(self.flatten_maze == 0) # states cardinality: walkable cells
        self.n_actions = 4 # actions cardinality: up, down, left, right

        # sampling initial state from a uniform distribution over all possible states
        self.terminal_state = np.random.choice(np.arange(self.n_states))
 
    def compute_prior_distribuitions(self):
        # compute prior distribuition w.r.t. initial state
        self.p_0 = np.zeros(self.n_states)
        self.p_0[self.initial_state] = 1
    
    def compute_transition_distribuition(self):
        # compute transition probability matrix given to the environment P(S'|S,A)
        self.p = np.zeros((self.n_states, self.n_states, self.n_actions)) # init

        for state in range(self.n_states):
            x, y = self.coordinates_from_state(state)

            # compute neighbors: state agent can go from current state in (x,y)
            neighbors_coord = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)] # up, down, left, right
            neighbors = [self.maze[coord] for coord in neighbors_coord]

            # compute probability to transited to neighbors from current state in (x,y)
            for neighbor, (y_neighbor, x_neighbor), action in zip(neighbors, neighbors_coord, [0, 1, 2, 3]):
                # action: [0, 1, 2, 3] -> [up, down, left, right]
                if neighbor == 0:
                    neighbor_state = self.state_from_coordinates(x_neighbor, y_neighbor)
                    self.p[neighbor_state, state, action] = 1
            
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
    
    def render(self):
        if self.window is None:
            # init pygame
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.maze_width*20, self.maze_height*20)
            )
        
        # draw maze
        self.window.fill((255, 255, 255))
        pix_square_size = 20
        for x in range(self.maze_width):
            for y in range(self.maze_height):
                if self.maze[y, x]:
                    pygame.draw.rect(
                        self.window,
                        (0, 0, 0),
                        pygame.Rect(
                            pix_square_size * x,
                            pix_square_size * y,
                            pix_square_size,
                            pix_square_size,
                        ),
                    )
        
        # draw terminal state
        def drawX(x,y):
            pygame.draw.lines(self.window, (255, 0, 0), True, [(x-10,y-10),(x+10,y+10)], 8)
            pygame.draw.lines(self.window, (255, 0, 0), True, [(x-10,y+10),(x+10,y-10)], 8)
        drawX(pix_square_size*self.terminal_state_coord[0]-10, pix_square_size*self.terminal_state_coord[1]+10)

        # draw initial state
        pygame.draw.rect(self.window, 
            (0, 100, 255), 
            pygame.Rect(
                pix_square_size * self.initial_state_coord[0],
                pix_square_size * self.initial_state_coord[1],
                pix_square_size,
                pix_square_size,
            ),
        )
        
        # display update
        pygame.display.update()