import numpy as np
from src.agents.agent import Agent

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]


class ModelAgent(Agent):
    # ricreo environment con modello approssimato
    def __init__(self, gamma, initial_state_coord):
        super().__init__(initial_state_coord=initial_state_coord)

        # initialize model agent
        self.gamma = gamma # discount factor    
        self.quality_function = dict() # Q := {(x,y): actions values (np.darray)}
        self.quality_function[initial_state_coord] = np.zeros(N_ACTIONS) # initialize quality function at initial state
    
    def reset_agent(self, initial_state_coord):
        self.__init__(
            gamma=self.gamma,
            initial_state_coord=initial_state_coord,
        )
    
    def __compute_reward_function(self, episode):
        """
        Compute cumulative reward from the episode
        episode = [(state, action, reward, next_state), ...]
        """
        T = len(episode) # number of steps to reach terminal state from current state
        if T == 1:
            return episode[0][2] # reward of the final step
        else:
            return sum([self.gamma**t * self.__compute_reward_function(episode[1:]) for t in range(T)]) # recursive call
    
