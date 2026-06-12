import numpy as np

from .tabular_agent import TabularAgent


class QLearning(TabularAgent):
    """
    Implementación el agente Q-Learning de aprendizaje por refuerzo, que utiliza el método de diferencias
    temporales, lo que implica que actualizan sus retornos estimados en cada paso del episodio, sin
    esperar a que terminen.

    Q-Learning es Off-Policy, aprende el valor de la mejor política posible (política objetivo),
    independientemente de lo que haga el agente (política de comportamiento).

    """
    def __init__(self, environment, epsilon, discount_factor, learning_rate, random_seed, epsilon_decay=False):
        super().__init__(environment, epsilon, random_seed)

        self.discount_factor = discount_factor
        self.episode_return = 0
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay

    def get_action(self, state: int):
        """
        Devuelve una acción a tomar por el agente, dependiendo del estado donde se encuentr
        :param state: estado actual
        :return: accion a tomar por el agente
        """
        return self._epsilon_greedy_policy(state)

    def on_episode_step(self, new_state, action, reward, t, episode_finished):
        """
        Llamado cuando un nuevo paso ha transcurrido en el episodio actual. En este caso,
        es cuando Q-Learning actualiza su tabla de retornos

        :param new_state: nuevo estado tras tomar la acción especificada
        :param action: acción tomada por el agente
        :param reward: recompensa obtenida tras tomar esa acción
        :param t: número de episodio actual
        :param episode_finished: si el episodio ha sido completado
        """

        self.episode.append((self.state, action, reward)) # Guardamos los valores de este paso para el registro de todo el episodio

        # Si ha terminado el episodio, el retorno objetivo es la recompensa obtenida
        # En caso contrario, obtenemos el retorno esperado máximo de la tabla Q para el siguiente estado (siguiendo política greedy de comportamiento, dado que es Off-Policy)
        # y calculamos el retorno objetivo como la recompensa inmediata mas retorno esperado por un factor de descuento.
        # Si el episodio ha terminado el retorno esperado es 0
        if episode_finished:
            target_return = reward
        else:
            expected_return = (not episode_finished) * np.max(self.Q[new_state])
            target_return = reward + self.discount_factor * expected_return

        estimated_return = self.Q[self.state][action] # Calcular retorno esperado para el estado actual y la acción que se tomó
        temporal_difference = target_return - estimated_return # Calcular las diferencias temporales
        self.Q[self.state][action] = estimated_return + self.learning_rate * temporal_difference # Actualizar tabla Q con las diferencias termporales aplicando una tasa de aprendizaje

        self.episode_return += reward
        self.state = new_state

        if self.epsilon_decay:
            self.epsilon = min(1.0, self.decay_factor / (t + 1))

    def on_episode_finished(self):
        """
        Llamado cuando un episodio ha sido completado
        """
        episode_reward = self.episode_return
        self.episode_return = 0
        self.episode = []  # Limpiar información del episodio
        return episode_reward