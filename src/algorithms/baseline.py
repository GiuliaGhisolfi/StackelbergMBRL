import numpy as np
from src.stackelberg_agent import StackelbergAgent
from src.algorithms.utils import compute_action_between_states

N_ACTIONS = 4
ACTIONS_LIST = [0, 1, 2, 3] # [up, down, left, right]
PROBABILITY_PENALIZATION_FACTOR = 0.1


class Baseline(StackelbergAgent):
    def __init__(self, gamma, initial_state_coord, transition_matrix_initial_state):
        super().__init__(gamma, initial_state_coord, transition_matrix_initial_state)

    def update_policy(self, previous_state_coord, previous_state_cardinality, transition_matrix):
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