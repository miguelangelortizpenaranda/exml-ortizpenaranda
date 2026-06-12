from abc import ABC, abstractmethod

import numpy as np
from gymnasium.spaces import Box


class Agent(ABC):
    """
    Clase abstracta que representa a un agente de aprendizaje por refuerzo,
    implementada por los agentes tabulares y aproximados
    """
    def __init__(self, environment, random_seed):
        self.name = self.__class__.__name__
        self.state = 0
        np.random.seed(random_seed)

    def set_initial_state(self, state):
        """
        Fija el estado inicial para el agente
        :param state: estado inicial procedente del entorno
        """
        self.state = state

    def get_name(self):
        """
        Devuelve el nombre del agente
        :return: nombre del agente
        """
        return self.name

    @abstractmethod
    def get_action(self, state: int):
        """
        Devuelve una acción a tomar por el agente, dependiendo del estado donde se encuentr
        :param state: estado actual
        :return: accion a tomar por el agente
        """
        pass

    @abstractmethod
    def on_episode_step(self, new_state, action, reward, t, episode_finished):
        """
        Llamado cuando un nuevo paso ha transcurrido en el episodio actual
        :param new_state: nuevo estado tras tomar la acción especificada
        :param action: acción tomada por el agente
        :param reward: recompensa obtenida tras tomar esa acción
        :param t: número de episodio actual
        :param episode_finished: si el episodio ha sido completado
        """
        pass

    @abstractmethod
    def on_episode_finished(self):
        """
        Llamado cuando un episodio ha sido completado. Es especialmente conveniente
        para algunos agentes como Monte Carlo
        """
        pass


