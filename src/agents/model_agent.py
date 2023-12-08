import numpy as np
from src.agents.agent import Agent

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]


class ModelAgent(Agent):
    # Model Agent: learns a model of the environment from experience
    def __init__(self, gamma, initial_state_coord, transition_matrix_initial_state):
        super().__init__(initial_state_coord=initial_state_coord)

        # initialize model
        self.gamma = gamma # discount factor
        self.agent_state = 0 # initial state
        
        self.transition_distribuition = [] # P(A|S) transition_distribuition[state] = probability distribution over actions

        self.next_state_function = dict()
        self.reward_function = dict()
        self.states_space = dict() # S: {(x,y): state number}
        self.values_function = dict() # V: {state: value np.darray}
        self.update_states_space(initial_state_coord)

        self.quality_function = dict() # Q: {state: quality np.darray}
        self.quality_function[initial_state_coord] = np.zeros(N_ACTIONS) 
    
    def step(self, action, reward, next_state_coord):
        self.update_states_space(state_coord=next_state_coord)
        next_state = self.states_space[next_state_coord]

        self.update_next_state_function(state=self.agent_state, action=action, next_state=next_state)
        self.update_reward_function(state=self.agent_state, action=action, reward=reward)

        self.agent_state = next_state
    
    def reset_agent(self, initial_state_coord, transition_matrix_initial_state):
        self.__init__(
            gamma=self.gamma, 
            initial_state_coord=initial_state_coord,
            transition_matrix_initial_state=transition_matrix_initial_state
        )

    def update_next_state_function(self, state, action, next_state):
        # S, A -> S': deterministica
        self.next_state_function[(state, action)] = next_state

    def update_reward_function(self, state, action, reward):
        # S, A -> R
        self.reward_function[(state, action)] = reward
    
    def update_states_space(self, state_coord):
        # state_coord from environment: (x,y) -> state in S
        if state_coord not in self.states_space.keys():
            s = len(self.states_space)
            self.states_space[state_coord] = s
            self.values_function[s] = np.zeros(N_ACTIONS)
