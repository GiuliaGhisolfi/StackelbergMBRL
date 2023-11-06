class Environment():

    def __init__(self):
        self.start = None # agent starting position
        self.goal = None # agent goal position
        self.agent_state = None
        self.maze = None # grafo
        pass

    def compute_reward(self, state, action):
        pass