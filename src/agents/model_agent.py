import numpy as np
from src.agent import Agent

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]


class ModelAgent(Agent):
    # ricreo environment con modello approssimato
    def __init__(self, gamma, initial_state_coord, transition_matrix_initial_state):
        super().__init__(gamma=gamma, initial_state_coord=initial_state_coord)

        # initialize model
        self.agent_state = 0 # initial state
        self.transition_distribuition = dict() # P(A|S)
        self.__update_actions_distribuition(transition_matrix_initial_state)
        self.next_state_function = dict()
        self.reward_function = dict()
        self.state_space = dict() # S
        self.__update_state_space(initial_state_coord)
    
    def step(self, action, reward, next_state_coord):
        self.__update_state_space(state_coord=next_state_coord)
        next_state = self.state_space[next_state_coord]

        self.__update_next_state_function(state=self.agent_state, action=action, next_state=next_state)
        self.__update_reward_function(state=self.agent_state, action=action, reward=reward)

        self.agent_state = next_state
    
    def __update_actions_distribuition(self, transition_matrix):
        # P(A|S)
        if self.agent_state not in self.transition_distribuition.keys():
            self.transition_distribuition[self.agent_state] = np.where(transition_matrix != 0)[0]
            self.transition_distribuition[self.agent_state] /= np.sum(self.transition_distribuition[self.agent_state])
        
    def __update_next_state_function(self, state, action, next_state):
        # S, A -> S': deterministica
        self.next_state_function[(state, action)] = next_state

    def __update_reward_function(self, state, action, reward):
        # S, A -> R
        self.reward_function[(state, action)] = reward
    
    def __update_state_space(self, state_coord):
        # state_coord from environment: (x,y) -> state in S
        if state_coord not in self.state_space.keys():
            self.state_space[state_coord] = self.agent_state
    
