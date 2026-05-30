import numpy as np

from agents.agent import Agent

class QLearning(Agent):

    def __init__(self, environment, epsilon, discount_factor, learning_rate, random_seed, epsilon_decay=False):
        super().__init__(environment, epsilon, random_seed)

        self.Q = np.zeros([self.n_states, self.n_actions])
        self.n_visits = np.zeros([self.n_states, self.n_actions])

        self.discount_factor = discount_factor
        self.factor = 1
        self.episode_return = 0
        self.learning_rate = learning_rate
        self.cumulative_error = 0
        self.epsilon_decay = epsilon_decay

    def get_action(self, state: int):
        return self._epsilon_greedy_policy(state)

    def update_episode_info(self, new_state, action, reward, t, episode_finished):
        """Update Q-value based on experience.
        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        expected_return = (not episode_finished) * np.max(self.Q[new_state]) # Get best return from the new state. Will be 0 if episode is finished
        target_return = reward + self.discount_factor * expected_return # Take into account current return and future return with a certain discount factor (Bellman equation)
        temporal_difference = target_return - self.Q[self.state][action] # Compute temporal differences using previous state and current state return
        estimated_return = self.Q[self.state][action]
        self.Q[self.state][action] = estimated_return + self.learning_rate * temporal_difference # Update Q table with the temporal difference applying a learning rate

        self.cumulative_error += temporal_difference
        self.episode_return += reward

        self.state = new_state

        if self.epsilon_decay:
            self.epsilon = min(1.0, 1000.0 / (t + 1))

    def update_post_episode(self):
        episode_reward = self.episode_return
        self.episode_return = 0
        self.cumulative_error = 0
        return episode_reward

    def getQ(self):
        return self.Q

    def getCurrentEpsilon(self):
        return self.epsilon