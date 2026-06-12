import numpy as np

from .tabular_agent import TabularAgent


class SARSA(TabularAgent):
    """
    Implementación el agente SARSA de aprendizaje por refuerzo, que utiliza el método de diferencias
    temporales, lo que implica que actualizan sus retornos estimados en cada paso del episodio, sin
    esperar a que terminen.

    SARSA es On-Policy, es decir,  aprende el valor de la política que realmente está siguiendo el agente para moverse.
    Actualiza su tabla Q usando la acción que realmente va a tomar después.
    """
    def __init__(self, environment, epsilon, discount_factor, learning_rate, random_seed, epsilon_decay=False):
        super().__init__(environment, epsilon, random_seed)

        self.discount_factor = discount_factor
        self.episode_return = 0
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.next_action = None  # Reseteamos para el nuevo episodio

    def get_action(self, state: int):
        """
        Devuelve una acción a tomar por el agente, dependiendo del estado donde se encuentr
        :param state: estado actual
        :return: accion a tomar por el agente
        """
        # Evita volver a obtener la acción de la política si ya la obtuvo previamente
        if self.next_action is not None:
            action = self.next_action
            self.next_action = None
            return action
        return self._epsilon_greedy_policy(state)

    def on_episode_step(self, new_state, action, reward, t, episode_finished):
        """
        Llamado cuando un nuevo paso ha transcurrido en el episodio actual. En este caso,
        es cuando SARSA actualiza su tabla de retornos

        :param new_state: nuevo estado tras tomar la acción especificada
        :param action: acción tomada por el agente
        :param reward: recompensa obtenida tras tomar esa acción
        :param t: número de episodio actual
        :param episode_finished: si el episodio ha sido completado
        """
        self.episode.append((self.state, action, reward))

        # Si ha terminado el episodio, el retorno objetivo es la recompensa obtenida
        # En caso contrario, obtenemos la siguiente acción en base a nuestra política
        # y comprobamos su retorno esperado en la tabla Q (en base a la siguiente acción a tomar en el nuevo estado), y su retorno objetivo
        # como recompensa inmediata mas ese retorno esperado menos un factor de descuento
        if episode_finished:
            target_return = reward
        else:
            self.next_action = self._epsilon_greedy_policy(new_state)
            expected_return = self.Q[new_state][self.next_action]
            target_return = reward + (self.discount_factor * expected_return)

        estimated_return = self.Q[self.state][action] # Calculamos retorno estimado en función al estado actual, y la acción que se tomó
        temporal_difference = target_return - estimated_return # Calcular diferencias temporales usando el retorno del futuro y el retorno actual
        self.Q[self.state][action] = estimated_return + self.learning_rate * temporal_difference # Actualizar la tabla Q con las diferencias temporales aplicando una tasa de aprendizaje

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
        self.next_action = None  # Reseteamos para el nuevo episodio
        self.episode = []  # Vaciamos info del episodio
        return episode_reward