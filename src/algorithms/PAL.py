import numpy as np
import json
from keras.losses import KLDivergence
from src.agents.model_agent import ModelAgent
from src.agents.policy_agent import PolicyAgent
from src.environment.environment import Environment
from src.algorithms.utils import stackelberg_nash_equilibrium

ACTION_LIST = [0, 1, 2, 3] # [up, down, left, right]

def compute_model_loss(transition_probability_model, transition_probability_data):
    # compute loss between transition_probability_model and transition_probability_data
    #FIXME: compute loss only on non-zero elements of transition_probability_data (q) 
    # and corresponding elements of transition_probability_model (p)
    # poi ricordarsi di sostituire in modo che in q tutte le righe sommino a 1
    y_true = transition_probability_model[~np.all(transition_probability_data == 0, axis=1)]
    y_pred = transition_probability_data[~np.all(transition_probability_data == 0, axis=1)]

    kl_divergence = compute_kl_divergence(
        y_true=transition_probability_model, 
        y_pred=transition_probability_data
        )
    
    loss = kl_divergence.sum() # sum over all actions
    return loss

def compute_kl_divergence(y_true, y_pred):
    # compute KL divergence between y_true and y_pred
    kl_divergence = KLDivergence()
    return kl_divergence(y_true, y_pred).numpy()


class PAL():
    # PAL: Policy As Leader Algorithm

    def __init__(self, learning_rate, n_environments, max_iterations_per_environment, n_episodes_per_iteration, 
        max_epochs_per_episode, maze_width, maze_height, gamma):

        self.n_environments = n_environments
        self.max_iterations_per_environment = max_iterations_per_environment
        self.n_episodes_per_iteration = n_episodes_per_iteration
        self.max_epochs_per_episode = max_epochs_per_episode
        self.lr = learning_rate # alpha

        # initalize environment
        self.env = Environment(
            maze_width=maze_width,
            maze_height=maze_height
            )

        # initialize model and policy agents
        self.model_agent = ModelAgent(
            gamma=gamma,
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :],
            initial_state_coord=self.env.initial_state_coord
            )
        self.policy_agent = PolicyAgent(
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :]
            )
    
    def train(self):
        for _ in range(self.n_environments):
            # train loop for the environment
            self.__train_loop_for_the_environment()

            #TODO: save policy in json file every n iterations, idk how many

            # initialize environment and model agent
            self.env.reset_environment()
            self.model_agent.reset_agent()

            # reset policy agent at initial state
            self.policy_agent.reset_at_initial_state(
                transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])

        # save final policy in json file
        json.dump(self.policy_agent.policy, open('../policy/policy.json', 'w'))
    
    def reset_at_initial_state(self):
        # reset agent's position in the environment
        self.env.reset()

        # reset agent state at initial state
        self.model_agent.agent_state = 0
        self.policy_agent.reset_at_initial_state(
            transition_matrix_initial_state=self.env.p[:, self.env.initial_state, :])
 
    def __train_loop_for_the_environment(self):
        for _ in range(self.max_iterations_per_environment):
            data_buffer = [] # list of episodes, each episode is a list of tuples (state, action, reward, next_state)

            # collect data executing policy in the environment
            for _ in range(self.n_episodes_per_iteration):
                episode = self.executing_policy()
                data_buffer.append(episode)
                self.reset_at_initial_state() # reset environment and agent state
            
            # build policy-specific model
            self.model_agent.transition_distribuition = self.__optimize_model(data_buffer)
            
            # improve policy
            self.policy_agent.policy = self.__improve_policy()

            #TODO: add stopping criterion == convergence at nash equilibrium
            if self.__check_stackelberg_nash_equilibrium():
                break
        
    def executing_policy(self):
        """
        Execute policy in the environment to collect data

        Args:
            agent (PolicyAgent)
            env (Environment)

        Returns:
            list os steps: [(state, action, reward, next_state), ...]
        """

        current_state = self.env.state
        current_state_coord = self.env.coordinates_from_state(self.env.state)
        action = self.policy_agent.fittizial_first_action
        episode = [] # init

        for _ in range(self.max_epochs_per_episode):
            # select action from policy
            action = self.policy_agent.compute_next_action(action=action, 
                        transition_matrix=self.env.p[:, current_state, :])

            # compute step in environment
            next_state, reward, _, _, _ = self.env.step(action) # netx_state: numeric

            # save step in episode
            episode.append((current_state_coord, action, reward, self.env.coordinates_from_state(next_state)))

            # update current state
            current_state = next_state
            current_state_coord = self.env.coordinates_from_state(next_state)
            
            # check if terminal state is reached
            if current_state == self.env.terminal_state:
                break
            
        return episode
    

    def __optimize_model(self, data_buffer):
        # build model given data_buffer
        # minimizzo KL div tra modello precedente (matrice di transizione P) e 
        # approx del modello ricavata dai dati (matrice Q di approx di P dagli episodi in data_buffer)
        loss = np.inf
        self.kl_divergence = []
        self.optimal_model = -1
        self.states_space = self.model_agent.states_space

        for i, episode in enumerate(data_buffer):
            q = self.__compute_transition_probability_from_episode(episode)
            p = self.__comunpute_transition_probability_from_model()
            local_loss = compute_model_loss(
                transition_probability_model=p,
                transition_probability_data=q)
            if local_loss < loss:
                loss = local_loss
                p = q
                self.optimal_model = i
            self.kl_divergence.append(local_loss) # in [0, +inf]

        return p
    
    def __compute_transition_probability_from_episode(self, episode):
        # compute transition probability from episode
        # episode = [(state, action, reward, next_state), ...]
        self.states_model_space_dim = len(self.states_space)
        
        for state, action, reward, next_state in episode:
            if state not in self.states_space.keys():
                self.states_model_space_dim += 1
                self.states_space[state] = self.states_model_space_dim-1
                
        q = np.zeros((self.states_model_space_dim, len(ACTION_LIST)))
        for state, action, reward, next_state in episode:
            q[self.states_space[state], action] += 1
            #TODO: update netx_state_function and reward_function ?
        q /= np.sum(q, axis=1)[:, None]
        q[np.isnan(q)] = 0 # if 0/0, then 0
        return q #FIXME: TUTTE LE RIGHE DI Q DEVONO SOMMARE A 1
    
    def __comunpute_transition_probability_from_model(self):
        # compute transition probability from model
        # transition_distribuition = {state: probability distribution over actions}
        p = np.zeros((self.states_model_space_dim, len(ACTION_LIST)))
        for state, actions_distribuition in self.model_agent.transition_distribuition.items():
            p[state, :] = actions_distribuition
        return p
    
    def __improve_policy(self):
        # improve policy given transition_distribuition
        gradient = self.__compute_cost_function_gradient()
        self.policy_agent.policy += self.lr * gradient

    def __compute_cost_function_gradient(self):
        #TODO
        pass

    def __check_stackelberg_nash_equilibrium(self):
        # compute stackelberg nash equilibrium
        # return True if the policy is at nash equilibrium, False otherwise
        # verify if I'm in a equilbrium, knowing all possible transition distribuition from the model and the policy

        equilibria = np.where(stackelberg_nash_equilibrium(
            leader_payoffs=-self.kl_divergence, 
            follower_payoffs=self.cost_function #FIXME: un elemento per ogni kl div computata, i.e. per ogni episodio
            ) != 0)[0]

        return self.optimal_model in equilibria
