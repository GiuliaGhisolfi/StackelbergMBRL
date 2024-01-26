import json
import numpy as np

###### STACKELBERG NASH EQUILIBRIUM FUNCTIONS ######
def check_stackelberg_nash_equilibrium(leader_payoffs, follower_payoffs, equilibrium_find):
    """
    Check if the policy is at Stackelberg-Nash equilibrium, 
    knowing all possible transition distributions from the model and the policy

    Returns:
        bool: True if the policy is at Nash equilibrium, False otherwise
    """
    follower_eq = np.argmax(follower_payoffs) # best response
    leader_eq = np.argmax(leader_payoffs)

    if follower_eq != leader_eq:
        return False
    return (leader_eq, follower_eq) == equilibrium_find

###### SAVE AND LOAD FUNCTIONS ######
def save_parameters(parameters_dict:dict, algorithm:str):
    # save parameters in json file
    with open(f'training_parameters/{algorithm}_parameters.json', 'w') as parameters_file:
        json.dump(parameters_dict, parameters_file)
    
def save_policy(policy, states_space, algorithm, environment_number):
    # save final policy and policy states space in json file
    with open(f'training_parameters/{algorithm}/policy/policy_env_{environment_number}.json', 'w') as policy_file:
        json.dump([row.tolist() for row in policy], policy_file)

    with open(f'training_parameters/{algorithm}/policy/states_space_env_{environment_number}.json', 
        'w') as states_space_file:
        json.dump({str(key): value.tolist() for key, value in states_space.items()}, states_space_file)

def save_metrics(metrics_dict, environment_number, algorithm):
    with open(f'training_parameters/{algorithm}/metrics/metrics_env_{environment_number}.json', 
        'w') as metrics_file:
        json.dump(metrics_dict, metrics_file)