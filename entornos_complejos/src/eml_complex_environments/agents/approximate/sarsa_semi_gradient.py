import numpy as np

from .approximate_agent import ApproximateAgent


class SARSASemiGradient(ApproximateAgent):
    """
    Implementa un agente utilizando el método de SARSA Semi-Gradiente, que utiliza, en lugar
    de una tabla que asocia estado-acción, una función de aproximación lineal, con una serie
    de pesos, para inferir los valores del retorno futuro.

    Para ello, utilizamos un wrapper del entorno que realiza un tile encoding, de forma que se consigan
    discretizar los estados del mismo, originalmente compuestos por componentes continuas, en distintos
    teselados que se solapan unos con otros. Estar en un estado concreto activará un conjunto de esas
    casillas que permitirán determinar qué características han de tenerse en cuenta
    para la aproximación lineal comentada.
    """
    def __init__(self, tile_coded_env, epsilon, discount_factor, learning_rate, random_seed, epsilon_decay=True, decay_factor=0.997):

        super().__init__(tile_coded_env, epsilon, random_seed)

        self.discount_factor = discount_factor
        self.initial_epsilon = epsilon
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.epsilon_decay = epsilon_decay
        self.tile_coded_env = tile_coded_env

        n_features = self.tile_coded_env.n_tilings * np.prod(self.tile_coded_env.bins) # Las features son tantas como número total de tiles
        self.weights = np.zeros((self.n_actions, n_features)) # Un vector de pesos de n_features por cada acción

        self.active_features = None
        self.next_action = None

        self.episode_return = 0.0
        self.episode_len = 0

    def set_initial_state(self, state):
        """
        Fija el estado inicial para el agente
        :param state: estado inicial procedente del entorno
        """
        self.active_features = self.tile_coded_env.get_last_active_features()
        self.next_action = None
        self.episode_len = 0

    def get_action(self, state):
        """
        Devuelve una acción a tomar por el agente, dependiendo del estado donde se encuentre.
        Recordemos que, al usar tile encoding, las casillas activas nos dan las características a activar,
        que podemos usar con la política de epsilon greedy para obtener una acción a realizar
        :param state: estado actual
        :return: accion a tomar por el agente
        """
        # Evita volver a obtener la acción de la política si ya la obtuvo previamente
        if self.next_action is not None:
            action = self.next_action
            self.next_action = None
            return action
        else:
            features = self.tile_coded_env.get_last_active_features()
            action = self._epsilon_greedy_policy(features)
            return action

    def on_episode_step(self, new_state, action, reward, t, episode_finished=False):
        """
        Llamado cuando un nuevo paso ha transcurrido en el episodio actual. En este caso,
        calcula el retorno actual tomando las features activas, y a la acción ejecutada.

        Obtiene las nuevas features tras la ejecución del paso, para calcular el retorno esperado,
        y al igual que en sarsa tradicional, se obtienen las diferencias temporales como
        la diferencia de ambos valores, aplicando una tasa de aprendizaje, y se actualiza el vector de pesos.

        :param new_state: nuevo estado tras tomar la acción especificada
        :param action: acción tomada por el agente
        :param reward: recompensa obtenida tras tomar esa acción
        :param t: número de episodio actual
        :param episode_finished: si el episodio ha sido completado
        """
        current_return = self._q_value(self.active_features, action, self.weights)
        new_features = self.tile_coded_env.get_last_active_features()

        if episode_finished:
            target_return = reward
            self.next_action = None
        else:
            self.next_action = self._epsilon_greedy_policy(new_features)
            next_q = self._q_value(new_features, self.next_action, self.weights)
            target_return = reward + self.discount_factor * next_q

        temporal_differences_error = target_return - current_return

        self.weights[action, self.active_features] += self.learning_rate * temporal_differences_error

        self.active_features = new_features
        self.episode_return += reward
        self.episode_len += 1


    def on_episode_finished(self):
        """
        Llamado cuando un episodio ha sido completado.
        Actualizamos también el valor de epsilon en caso de estar usando un epsilon decay
        """
        total_return = self.episode_return
        self.episode_return = 0.0

        if self.epsilon_decay:
            self.epsilon = max(0.01, self.epsilon * self.decay_factor)

        return total_return

    def get_model(self):
        """
        Obtiene los pesos calculados para el agente
        """
        return self.weights

    def get_current_epsilon(self):
        """
        Devuelve el valor actual de epsilon
        """
        return self.epsilon

    def get_episode_len(self):
        """
        Devuelve el número de episodio actual
        """
        return self.episode_len

    def get_env(self):
        """
        Obtiene el entorno que utiliza esta agente
        """
        return self.tile_coded_env

    def _q_value(self, active_features, a, weights):
        """
        Calcula q(s,a) como la suma de los pesos para los índices activos.

        :param active_features: lista de índices de features activas para el estado s.
        :param a: acción seleccionada.
        :param weights: matriz de pesos de dimensiones [n_features, n_actions].

        :returns: valor aproximado de Q(s,a).
        """
        return weights[a, active_features].sum()

    def _epsilon_greedy_policy(self, state_features):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        q_values = self.weights[:, state_features].sum(axis=1) # evaluamos q(s,a) para todas las posibles acciones
        # Elige una acción y rompe empates cuando todos son 0
        return np.random.choice(np.flatnonzero(q_values == q_values.max()))
