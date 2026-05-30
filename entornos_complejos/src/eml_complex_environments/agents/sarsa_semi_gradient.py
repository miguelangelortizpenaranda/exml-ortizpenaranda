import numpy as np
from agents.agent import Agent


class SARSASemiGradient(Agent):

    def __init__(self, environment, epsilon, discount_factor, learning_rate, random_seed, epsilon_decay,
                 feature_dim=None):
        super().__init__(environment, random_seed)

        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate  # Tu parámetro Alpha (\alpha) de aproximación
        self.epsilon_decay = epsilon_decay

        # Si no se define la dimensión de características, asumimos espacio discreto (One-Hot)
        self.feature_dim = feature_dim if feature_dim is not None else self.n_states

        # Inicializamos los pesos (w). Tendremos un vector de pesos para cada acción.
        # Dimensión: (num_acciones, num_características)
        self.w = np.zeros((self.n_actions, self.feature_dim))

        self.state = 0
        self.episode_reward = 0.0

        # Buffer crucial para SARSA: Guarda la acción precalculada para el siguiente paso
        self.next_action = None

        # Acciones estándar del entorno

    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

    def _get_features(self, state):
        """
        Extractor de características x(s).
        Soporta entornos discretos (ints) y continuos (arrays como MountainCar).
        """
        if isinstance(state, (int, np.integer)):
            # Entorno discreto: Vector One-Hot del estado
            x = np.zeros(self.feature_dim)
            x[state] = 1.0
            return x
        else:
            # Entorno continuo: El estado ya es un vector de características numéricas
            return np.array(state, dtype=float)

    def _get_q_value(self, state, action):
        """Calcula q^ (s, a, w) = w_a^T * x(s)"""
        x = self._get_features(state)

        # Inicialización perezosa por si el entorno continuo varía su tamaño de vector
        if x.shape[0] != self.w.shape[1]:
            self.feature_dim = x.shape[0]
            self.w = np.zeros((self.n_actions, self.feature_dim))

        return np.dot(self.w[action], x)

    def get_action(self, state: int):
        # Si ya elegimos la acción en el update del paso anterior (esencia de SARSA), la usamos
        if self.next_action is not None:
            action = self.next_action
            self.next_action = None  # Vaciamos el buffer
            return action

        # Si es el primer paso del episodio, calculamos de cero
        return self._epsilon_greedy_policy(state)

    def update_episode_info(self, new_state, action, reward, t, episode_finished=False):
        self.episode_reward += reward

        # 1. Obtener el valor Q estimado del estado y acción actuales
        q_current = self._get_q_value(self.state, action)
        x_s = self._get_features(self.state)

        # 2. Calcular el Target de la ecuación de Bellman
        if episode_finished:
            target = reward
            self.next_action = None
        else:
            # SARSA On-Policy: Elegimos la acción del Siguiente Estado usando la política actual
            self.next_action = self._epsilon_greedy_policy(new_state)
            q_next = self._get_q_value(new_state, self.next_action)
            target = reward + self.discount_factor * q_next

        # 3. Calcular el error de Diferencia Temporal (TD Error)
        td_error = target - q_current

        # 4. Actualización Semi-Gradiente:
        # w_a <- w_a + learning_rate * td_error * \nabla_w q^(s, a, w)
        # Nota: Como q^ es lineal (w^T * x), el gradiente respecto a w_a es simplemente x(s)
        self.w[action] += self.learning_rate * td_error * x_s

        # Avanzar el estado del agente
        self.state = new_state

        if self.epsilon_decay:
            self.epsilon = min(1.0, 1000.0 / (t + 1))

    def update_post_episode(self):
        # Retornamos la recompensa acumulada para que tu gráfica estadística funcione igual
        total_reward = self.episode_reward
        self.episode_reward = 0.0
        return total_reward

    def getQ(self):
        """ Reconstruye una Tabla Q tradicional si el entorno es discreto """
        try:
            Q_table = np.zeros((self.n_states, self.n_actions))
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    Q_table[s, a] = self._get_q_value(s, a)
            return Q_table
        except AttributeError:
            # Si estás en un entorno continuo (sin n_states fijo), devolvemos los pesos puros
            return self.w

    def getCurrentEpsilon(self):
        return self.epsilon

    def _random_epsilon_greedy_policy(self, state):
        pi_A = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions

        # Calculamos los valores Q dinámicamente usando el aproximador lineal
        q_values = [self._get_q_value(state, a) for a in range(self.n_actions)]
        best_action = np.argmax(q_values)

        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    def _epsilon_greedy_policy(self, state):
        pi_A = self._random_epsilon_greedy_policy(state)
        return np.random.choice(np.arange(self.n_actions), p=pi_A)