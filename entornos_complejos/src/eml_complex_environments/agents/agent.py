from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):

    def __init__(self, environment, random_seed):
        self.n_states = environment.observation_space.n
        self.n_actions = environment.action_space.n
        self.state = 0
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


