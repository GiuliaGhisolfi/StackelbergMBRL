from maze import Maze
from environment import Environment
from agent import ModelAgent, PolicyAgent
from algorithms.MAL import MAL
from algorithms.PAL import PAL

class MazeSolver(Maze):

    def __init__(self, algorithm, size_x, size_y, gamma, n_episodes_per_iteration, alpha):
        # alpha: regularization parameter 
        super().__init__(size_x=size_x, size_y=size_y)
        self.algorithm = algorithm
        # self.maze, self.start, self.goal: inherited from Maze
        self.env = Environment()
        self.model_agent = ModelAgent(env=self.env, gamma=gamma)
        self.policy_agent = PolicyAgent(env=self.env)

        #TODO: contiunue here