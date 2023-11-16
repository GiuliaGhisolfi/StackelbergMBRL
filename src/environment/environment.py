import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import spy
import pygame
from matrix_mdp.envs import MatrixMDPEnv
from src.environment.maze import Maze

SCREEN_WIDTH_MAX = 1920
SCREEN_HEIGHT_MAX = 1080
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class Environment(MatrixMDPEnv):
    
    def __init__(self, maze_width, maze_height, block_pixel_size=None):
        # initializa maze
        print('Initialize maze')
        self.maze_environment = Maze(size_x=int(maze_width/2), size_y=int(maze_height/2))
        
        # maze parameters
        self.maze_width = int(maze_width/2) * 2 + 1
        self.maze_height = int(maze_height/2) * 2 + 1
        self.maze = self.maze_environment.blocks
        self.flatten_maze = self.maze_environment.blocks.flatten().reshape(1, -1)
        print('Maze created')
        print('Maze size: {}x{}'.format(self.maze_width, self.maze_height))

        # set block pixel size
        if block_pixel_size is None:
            self.block_pixel_size = int(max(SCREEN_WIDTH/self.maze_width, SCREEN_HEIGHT/self.maze_height))
        elif (self.maze_width*block_pixel_size > SCREEN_WIDTH_MAX or 
            self.maze_height*block_pixel_size > SCREEN_HEIGHT_MAX):
            self.block_pixel_size = int(min(SCREEN_WIDTH_MAX/self.maze_width, SCREEN_HEIGHT_MAX/self.maze_height))
        else:
            self.block_pixel_size = block_pixel_size
        # init pygame window
        self.window = None

        # initialize initial and terminal state
        print('Compute initial and terminal state')
        self.compute_terminal_states()
        self.compute_initial_states()
        print('Initial state: {}'.format(self.initial_state_coord))

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
                (self.maze_width * self.block_pixel_size, self.maze_height * self.block_pixel_size)
            )
        # draw maze
        self.window.fill((255, 255, 255))
        for x in range(self.maze_width):
            for y in range(self.maze_height):
                if self.maze[y, x]:
                    pygame.draw.rect(
                        self.window,
                        (0, 0, 0),
                        pygame.Rect(
                            self.block_pixel_size * x,
                            self.block_pixel_size * y,
                            self.block_pixel_size,
                            self.block_pixel_size,
                        ),
                    )
        
        # draw terminal state
        line_width = int(self.block_pixel_size/5)
        def drawX(x,y):
            pygame.draw.lines(
                self.window, 
                (255, 0, 0), 
                True, 
                [(x + line_width, y + line_width), 
                (x + self.block_pixel_size - 1 - line_width, y + self.block_pixel_size - 1 - line_width)], 
                line_width
            )
            pygame.draw.lines(
                self.window, 
                (255, 0, 0), 
                True, 
                [(x + line_width, y + self.block_pixel_size - 1 - line_width),
                (x + self.block_pixel_size - 1 - line_width, y + line_width)], 
                line_width
            )
        drawX(self.terminal_state_coord[0] * self.block_pixel_size, 
            self.terminal_state_coord[1] * self.block_pixel_size)

        # draw initial state
        pygame.draw.rect(self.window,
            (0, 0, 0), 
            pygame.Rect(
                (self.block_pixel_size * self.initial_state_coord[0] + line_width),
                (self.block_pixel_size * self.initial_state_coord[1] + line_width),
                (self.block_pixel_size - 2 * line_width),
                (self.block_pixel_size - 2 * line_width),
            ),
        )
