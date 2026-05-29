import numpy as np

from agents.agent import Agent

class MonteCarloOnPolicy(Agent):

    def __init__(self, environment, epsilon, discount_factor):
        self.n_states = environment.observation_space.n
        self.n_actions = environment.action_space.n
        self.Q = np.zeros([self.n_states, self.n_actions])
        self.n_visits = np.zeros([self.n_states, self.n_actions])
        self.epsilon = epsilon
        self.discount_factor = discount_factor

        self.factor = 1
        self.state = 0
        self.episode = []

    # actions
    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

    def get_action(self, state: int):
        return self._epsilon_greedy_policy(state)

    def update_episode_info(self, new_state, action, reward):
        self.episode.append((self.state, action, reward))
        self.factor *= self.discount_factor
        self.state = new_state

    def update(self):
        # Esto es un disparate ¿por qué?
        # Porque hay que usar el retorno por cada uno de los episodios pasados (t), no uno total!!
        # También recorre el episodio hacia delante, cuando normalmente es hacia atrás
        # for (state, action) in episode:
        #     n_visits[state, action] += 1.0
        #     alpha = 1.0 / n_visits[state, action]
        #     Q[state, action] += alpha * (result_sum - Q[state, action])

        #gamma = 0.99
        current_g = 0.0  # Inicializamos el retorno en el estado terminal (es 0.0)
        cummulative_g = 0.0

        for state, action, reward in reversed(self.episode):
            # El retorno actual es la recompensa inmediata + el retorno futuro descontado
            current_g = reward + self.factor * current_g
            cummulative_g += current_g
            # Actualizamos el contador de visitas para esta pareja estado-acción
            self.n_visits[state, action] += 1.0
            # Calculamos el alfa dinámico (la media exacta de todas las visitas de la historia)
            alpha = 1.0 / self.n_visits[state, action]
            # Actualizamos la Tabla Q con el retorno G correcto para este momento del tiempo
            self.Q[state, action] += alpha * (current_g - self.Q[state, action])

        # Al terminar de aprender, vaciamos el episodio para la siguiente partida
        self.episode = []
        return cummulative_g

    def get_episode_len(self):
        return len(self.episode)

    def getQ(self):
        return self.Q

    # Política epsilon-soft. Se usa para el entrenamiento
    def _random_epsilon_greedy_policy(self, state):
        pi_A = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    # Política epsilon-greedy a partir de una epsilon-soft
    def _epsilon_greedy_policy(self, state):
        pi_A = self._random_epsilon_greedy_policy(state)
        return np.random.choice(np.arange(self.n_actions), p=pi_A)

