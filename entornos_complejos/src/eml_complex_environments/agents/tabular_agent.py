from abc import ABC, abstractmethod

import numpy as np

from agents.agent import Agent

class TabularAgent(Agent):

    def __init__(self, environment, epsilon, random_seed):
        super().__init__(environment, random_seed)
        self.epsilon = epsilon
        self.Q = np.zeros([self.n_states, self.n_actions])

    @abstractmethod
    def getQ(self):
        pass

    @abstractmethod
    def getCurrentEpsilon(self):
        pass

    def print_q_table(self):
        # @title Tabla de valores Q
        #LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
        print("Valores Q para cada estado:\n", self.Q)

    def print_pi_star_from_q(self, env):
        # Política Greedy a partir de los valones Q. Se usa para mostrar la solución.
        done = False
        pi_star = np.zeros([env.observation_space.n, env.action_space.n])
        state, info = env.reset()  # start in top-left, = 0
        actions = ""
        while not done:
            action = np.argmax(self.Q[state, :])
            actions += f"{action}, "
            pi_star[state, action] = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        print("Política óptima obtenida\n", pi_star, f"\n Acciones {actions} \n Para el siguiente grid\n", env.render())
        print()

    # Epsilon-soft policy for training
    def _random_epsilon_greedy_policy(self, state):
        pi_A = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    # Epsilon-greedy policy derived from epsilon_soft
    def _epsilon_greedy_policy(self, state):
        pi_A = self._random_epsilon_greedy_policy(state)
        return np.random.choice(np.arange(self.n_actions), p=pi_A)