import numpy as np
from src.agents.agent import Agent

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]
PROBABILITY_PENALIZATION_FACTOR = 0.1


def compute_action_between_states(state_from_coord, state_to_coord):
    if state_from_coord[1] - 1 == state_to_coord[1]:
        return 0 # up
    elif state_from_coord[1] + 1 == state_to_coord[1]:
        return 1 # down
    elif state_from_coord[0] - 1 == state_to_coord[0]:
        return 2 # left
    elif state_from_coord[0] + 1 == state_to_coord[0]:
        return 3 # right
    return None # error

class Baseline(Agent):
    def __init__(self, initial_state_coord, transition_matrix_initial_state):
        super().__init__(initial_state_coord)
    
        self.model_states_space = dict() # discrete: {(x,y): [possible actions]}
        self.__initialize_policy(np.where(transition_matrix_initial_state != 0)[1])
        self.__update_states_space(transition_matrix_initial_state)
    
    def __initialize_policy(self, actions_list_initial_state):
        self.policy = dict() # {(x,y): [action's probability]}

        actions_probability_list = np.zeros(N_ACTIONS)
        actions_probability_list[actions_list_initial_state] = 1/len(actions_list_initial_state)

        self.policy[self.agent_state_coord] = actions_probability_list
    
    def take_action(self):
        self.action = np.random.choice(ACTIONS_LIST, p=self.policy[self.agent_state_coord])
        return self.action
    
    def update_agent_parameters(self, next_state_coord, previous_state_coord, previous_state_cardinality, 
        transition_matrix, reached_terminal_state):
        self.__update_agent_state(next_state_coord)

        if not reached_terminal_state:
            self.__update_states_space(transition_matrix)
            self.__update_policy(previous_state_coord, previous_state_cardinality, transition_matrix)

    def __update_agent_state(self, next_state_coord):
        # update current state and path
        self.agent_state_coord = next_state_coord
        self.path.append(next_state_coord)
    
    def __update_states_space(self, transition_matrix):
        #transition_matrix = env.p[:, state, :]
        # state as a tuple (x,y)
        if self.agent_state_coord not in self.model_states_space.keys():
            self.model_states_space[self.agent_state_coord] = np.where(transition_matrix != 0)[1] # possible actions from state

    def __update_policy(self, previous_state_coord, previous_state_cardinality, transition_matrix):
        update_policy = False

        if self.agent_state_coord in self.policy.keys():
            if len(np.where(transition_matrix != 0)[1]) > 1:
                actions_probability_list = self.policy[self.agent_state_coord]
                actions_probability_list[np.where(transition_matrix[previous_state_cardinality, :] != 0)[0]
                    ] *= PROBABILITY_PENALIZATION_FACTOR
                update_policy = True
        else:
            # initialize policy for current state
            actions_probability_list = np.zeros(N_ACTIONS)

            # compute policy as uniform distribution over possible actions
            actions_list_state = np.where(transition_matrix != 0)[1]
            actions_probability_list[actions_list_state] = 1/len(actions_list_state)
            actions_probability_list[np.where(transition_matrix[previous_state_cardinality, :] != 0)[0]
                ] *= PROBABILITY_PENALIZATION_FACTOR
            
            update_policy = True
    
        if update_policy:
            # normalize probability distribution
            actions_probability_list = actions_probability_list / np.sum(actions_probability_list)
            # update policy
            self.policy[ self.agent_state_coord] = actions_probability_list
        
        # check if agent is in a blind corridor 
        if len(np.where(transition_matrix != 0)[1]) == 1:
            next_state_corridor_coord = previous_state_coord
            update_policy = False

            for start_blind_corridor_coord in reversed(self.path[:-1]):
                if  np.where(self.policy[start_blind_corridor_coord] != 0)[0].shape[0]  > 2:
                    # find action that leads from start_blind_corridor to next_state_corridor
                    action = compute_action_between_states(start_blind_corridor_coord, next_state_corridor_coord)

                    # set probability of action to 0
                    actions_probability_list = self.policy[start_blind_corridor_coord]
                    actions_probability_list[action] = 0
                    update_policy = True
                    break
                next_state_corridor_coord = start_blind_corridor_coord 
        
            if update_policy:
                # normalize probability distribution
                actions_probability_list = actions_probability_list / np.sum(actions_probability_list)
                # update policy
                self.policy[start_blind_corridor_coord] = actions_probability_list
            
                # update policy for all states in the blind corridor
                state_from_coord = self.agent_state_coord
                for state_to_coord in reversed(self.path):
                    action = compute_action_between_states(state_from_coord, state_to_coord)

                    # set probability of action to 1
                    actions_probability_list = np.zeros(N_ACTIONS)
                    actions_probability_list[action] = 1

                    # update policy
                    self.policy[state_from_coord] = actions_probability_list
                    
                    if state_to_coord == start_blind_corridor_coord:
                        break
                    state_from_coord = state_to_coord