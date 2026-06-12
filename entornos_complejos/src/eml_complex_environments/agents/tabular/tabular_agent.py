from abc import ABC, abstractmethod

import numpy as np

from ..agent import Agent

class TabularAgent(Agent):
    """
    Clase que implementa un agente tabular. Entre otros datos,
    contiene su tabla Q, e implementa los algoritmos de epsilon-greedy y epsilon-soft
    """
    def __init__(self, environment, epsilon, random_seed, decay_factor=1000):
        super().__init__(environment, random_seed)
        self.n_states = environment.observation_space.n
        self.n_actions = environment.action_space.n

        self.epsilon = epsilon
        self.Q = np.zeros([self.n_states, self.n_actions])
        self.episode = []
        self.decay_factor = decay_factor

    def get_q_table(self):
        """
        Devuelve la tabla Q construída hasta ahora por el agente
        :return: tabla que asocia pares estado-acción con retorno esperado de los mismos
        """
        return self.Q

    def get_current_epsilon(self):
        """
        Devuelve el valor de epsilon actual del agente
        :return: epsilon actual del agente
        """
        return self.epsilon

    def get_episode_len(self):
        """
        Obtiene la longitud (pasos) que han transcurrido hasta ahora en el episodio actual para el agente
        :return: número de pasos
        """
        return len(self.episode)

    def print_q_table(self):
        """
        Imprime la tabla Q construida hasta ahora por el agente
        """
        print("Valores Q para cada estado:\n", self.Q)

    def _random_epsilon_greedy_policy(self, state):
        """
        Política de epsilon soft, que permite elegir de entre las mejores acciones
        posibles, en base a la tabla Q, pero con una probabilidad epsilon de elegir una acción aleatoria
        :param state: estado actual sobre el que elegir la acción
        :return: array de probabilidades. Para cada acción, probabilidad de ser elegida en ese estado
        """
        pi_A = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        #best_action = np.argmax(self.Q[state])
        best_actions = np.flatnonzero(self.Q[state] == self.Q[state].max()) # Choose all possible best actions instead of the first
        best_action = np.random.choice(best_actions)
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    def _epsilon_greedy_policy(self, state):
        """
        Devuelve una acción concreta, dado un estado y la política epsilon soft.
        :param state: estado actual sobre el que elegir la acción
        :return: acción elegida por el agente
        """
        pi_A = self._random_epsilon_greedy_policy(state)
        return np.random.choice(np.arange(self.n_actions), p=pi_A)