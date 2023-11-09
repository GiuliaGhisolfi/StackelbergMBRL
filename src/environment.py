import numpy as np
import pygame
from matrix_mdp import MatrixMDPEnv
from src.maze import Maze

class Environment(MatrixMDPEnv):
    
    def __init__(self, maze_width, maze_height):
        self.maze_width = maze_width
        self.maze_height =  maze_height

        # initializa maze
        maze = Maze(size_x=int(self.maze_width/2), size_y=int(self.maze_height/2))
        self.maze = maze.blocks
        self.flatten_maze = maze.blocks.flatten().reshape(1, -1)

        p_0, p = self.compute_prior_probability()
        r = self.compute_reward_function()

        super.__init__(p_0=p_0, p=p, r=r, render_mode='human')
    
    def compute_initial_distribuitions(self):
        self.n_states =  np.sum(self.flatten_maze == 0) # states cardinality: walkable cells
        self.n_actions = 4 # actions cardinality: up, down, left, right

        self.p_0 = (np.ones(self.flatten_blocks.shape[1]) - self.flatten_blocks) / self.n_states # prior distribution

        # compute transition probability matrix given to the environment P(S'|S,A)
        self.p = self.compute_transition_distribuition()
    
    def compute_transition_distribuition(self):
        self.p = np.ones((self.n_states, self.n_states, self.n_actions)) / self.n_states # init
        for y in range(1, self.maze_height):
            for x in range(1, self.maze_width):
                den = np.sum(self.maze[x, y-1], self.maze[x, y+1], self.maze[x-1, y], self.maze[x+1, y])
                
        pass
    
    def compute_reward_function(self):
        pass