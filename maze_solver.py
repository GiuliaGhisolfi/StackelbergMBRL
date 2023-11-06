from maze import Maze

class MazeSolver(Maze):

    def __init__(self, algorithm, size_x, size_y, gamma):
        super().__init__(size_x=size_x, size_y=size_y)
        pass

    def PAL(self, policy, model, env):
        data_buffer = [] # list of tuples (state, action, reward, next_state)
        # collect data executing policy in the environment
        # build policy-specific model
        # improve policy
        return policy, model
    
    def MAL(self, policy, model, env):
        data_buffer = []
        # optimize policy massimizing reward given the model
        # collect data executing optimized policy in the environment
        # improve model
        return policy, model