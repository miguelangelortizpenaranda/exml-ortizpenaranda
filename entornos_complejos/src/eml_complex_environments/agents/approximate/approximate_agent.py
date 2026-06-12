from abc import abstractmethod

import numpy as np
from gymnasium.spaces import MultiDiscrete

from ..agent import Agent

class ApproximateAgent(Agent):
    """
    Clase que implementa un agente aproximado
    """
    def __init__(self, environment, epsilon, random_seed, decay_factor=1000):
        super().__init__(environment, random_seed)
        #self.n_states = environment.observation_space.n
        self.n_actions = environment.action_space.n
        self.epsilon = epsilon
        self.decay_factor = decay_factor

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_env(self):
        pass

    def get_current_epsilon(self):
        """
        Devuelve el valor de epsilon actual del agente
        :return: epsilon actual del agente
        """
        return self.epsilon
