import numpy as np

from agents.agent import Agent

class QLearning(Agent):

    def __init__(self, environment, epsilon, discount_factor, learning_rate):
        self.n_states = environment.observation_space.n
        self.n_actions = environment.action_space.n
        self.Q = np.zeros([self.n_states, self.n_actions])
        self.n_visits = np.zeros([self.n_states, self.n_actions])
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.factor = 1
        self.episode_return = 0
        self.state = 0
        self.learning_rate = learning_rate
        self.cumulative_error = 0

    def get_action(self, state: int):
        return self._epsilon_greedy_policy(state)

    def update_episode_info(self, new_state, action, reward):
        """Update Q-value based on experience.
        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = np.max(self.Q[new_state])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.Q[self.state][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.Q[self.state][action] = self.Q[self.state][action] + self.learning_rate * temporal_difference

        self.cumulative_error += temporal_difference

    def update(self):
        episode_error = self.cumulative_error
        self.cumulative_error = 0
        return episode_error

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


    def get_episode_len(self):
        return len(self.episode)

    def getQ(self):
        return self.Q