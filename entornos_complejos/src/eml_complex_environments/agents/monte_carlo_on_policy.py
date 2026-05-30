import numpy as np

from agents.tabular_agent import TabularAgent


class MonteCarloOnPolicy(TabularAgent):

    def __init__(self, environment, epsilon, discount_factor, random_seed, epsilon_decay):
        super().__init__(environment, epsilon, random_seed)

        self.n_visits = np.zeros([self.n_states, self.n_actions])
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.accumulated_return = 0
        self.episode = []

    # actions
    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

    def get_action(self, state: int):
        return self._epsilon_greedy_policy(state)

    def update_episode_info(self, new_state, action, reward, t, episode_finished):
        self.episode.append((self.state, action, reward))
        self.state = new_state
        if self.epsilon_decay:
            self.epsilon = min(1.0, 1000.0 / (t + 1))

    def update_post_episode(self):
        # Esto es un disparate ¿por qué?

        # --> Porque hay que usar el retorno por cada uno de los episodios pasados (t), no uno total!!
        # --> También recorre el episodio hacia delante, cuando normalmente es hacia atrás

        # for (state, action) in episode:
        #     n_visits[state, action] += 1.0
        #     alpha = 1.0 / n_visits[state, action]
        #     Q[state, action] += alpha * (result_sum - Q[state, action])

        #gamma = 0.99
        current_return = 0.0  # Terminal state (current return = 0)

        for state, action, reward in reversed(self.episode):
            current_return = reward + self.discount_factor * current_return # Current return equals to immediate reward + future return with a discount factor applied
            self.n_visits[state, action] += 1.0 # Update visit counter for this state-action pair
            learning_rate = 1.0 / self.n_visits[state, action] # Compute learning rate as inverse of visit count for this state-action pair
            expected_return = self.Q[state, action]
            self.Q[state, action] += learning_rate * (current_return - expected_return) # Update return table with the proper return for this step

        total_episode_reward = sum(reward for _, _, reward in self.episode)
        self.episode = [] # Clean episode info for next episode

        return total_episode_reward

    def get_episode_len(self):
        return len(self.episode)

    def getQ(self):
        return self.Q

    def getCurrentEpsilon(self):
        return self.epsilon

