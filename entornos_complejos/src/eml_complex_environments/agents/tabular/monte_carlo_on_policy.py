import numpy as np

from .tabular_agent import TabularAgent


class MonteCarloOnPolicy(TabularAgent):
    """
    Implementación del agente Monte Carlo On-Policy. El agente tiene en cuenta una única política, que es
    la que dictamina sus movimientos por el entorno, en base a la tabla Q. El agente calcula sus retornos
    cuando llega al final de un episodio.
    """
    def __init__(self, environment, epsilon, discount_factor, random_seed, epsilon_decay):
        super().__init__(environment, epsilon, random_seed)

        self.n_visits = np.zeros([self.n_states, self.n_actions])
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.accumulated_return = 0

    def get_action(self, state: int):
        """
        Devuelve una acción a tomar por el agente, dependiendo del estado donde se encuentr
        :param state: estado actual
        :return: accion a tomar por el agente
        """
        return self._epsilon_greedy_policy(state)

    def on_episode_step(self, new_state, action, reward, t, episode_finished):
        """
        Llamado cuando un nuevo paso ha transcurrido en el episodio actual
        :param new_state: nuevo estado tras tomar la acción especificada
        :param action: acción tomada por el agente
        :param reward: recompensa obtenida tras tomar esa acción
        :param t: número de episodio actual
        :param episode_finished: si el episodio ha sido completado
        """
        self.episode.append((self.state, action, reward))
        self.state = new_state
        if self.epsilon_decay:
            self.epsilon = min(1.0, self.decay_factor / (t + 1))

    def on_episode_finished(self):
        """
        Llamado cuando un episodio ha sido completado. Itera de forma inversa
        por todos los pasos de un episodio hacia atrás, y calcula los valores para cada
        par estado-acción por los que ha pasado en su tabla de retornos.
        """

        # Esto es un disparate ¿por qué?

        # --> Porque hay que usar el retorno por cada uno de los episodios pasados (t), no uno total
        # --> También recorre el episodio hacia delante, cuando normalmente es hacia atrás

        # for (state, action) in episode:
        #     n_visits[state, action] += 1.0
        #     alpha = 1.0 / n_visits[state, action]
        #     Q[state, action] += alpha * (result_sum - Q[state, action])

        current_return = 0.0  # Estado terminal

        for state, action, reward in reversed(self.episode):
            current_return = reward + self.discount_factor * current_return # El retorno actual equivale a la recompensa mas el retorno futuro con un facttor de descuento
            self.n_visits[state, action] += 1.0 # Actualizar contador de visitas para este par estado-acción
            learning_rate = 1.0 / self.n_visits[state, action] # La tasa de aprendizaje es la inversa de las visitas que ha recibido este par estado-acción
            expected_return = self.Q[state, action]
            self.Q[state, action] += learning_rate * (current_return - expected_return) # Actualizar tabla de retornos con el nuevo retorno para este par estado-acción

        total_episode_reward = sum(reward for _, _, reward in self.episode) # Calculamos la suma total de recompensas en este episodio
        self.episode = [] # Limpiar información del episodio

        return total_episode_reward



