from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):

    def __init__(self, environment, epsilon, random_seed):
        self.n_states = environment.observation_space.n
        self.n_actions = environment.action_space.n
        self.epsilon = epsilon
        self.state = 0
        self.Q = np.zeros([self.n_states, self.n_actions])
        np.random.seed(random_seed)

    def set_initial_state(self, state):
        self.state = state

    @abstractmethod
    def get_action(self, state: int):
        pass

    @abstractmethod
    def update_episode_info(self, new_state, action, reward, t, episode_finished):
        pass

    @abstractmethod
    def update_post_episode(self):
        pass

    @abstractmethod
    def getQ(self):
        pass

    @abstractmethod
    def getCurrentEpsilon(self):
        pass

    # Epsilon-soft policy for training
    def _random_epsilon_greedy_policy(self, state):
        pi_A = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    # Epsilon-greedy policy derived from epsilon_soft
    def _epsilon_greedy_policy(self, state):
        pi_A = self._random_epsilon_greedy_policy(state)
        return np.random.choice(np.arange(self.n_actions), p=pi_A)