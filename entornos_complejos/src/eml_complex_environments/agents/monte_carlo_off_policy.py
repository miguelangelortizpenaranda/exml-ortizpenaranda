import numpy as np

from agents.tabular_agent import TabularAgent

class MonteCarloOffPolicy(TabularAgent):

    def __init__(self, environment, epsilon, discount_factor, random_seed, epsilon_decay):
        super().__init__(environment, epsilon, random_seed)

        self.C = np.zeros([self.n_states, self.n_actions]) # Importances matrix

        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        self.accumulated_return = 0
        self.episode = []

    def get_action(self, state: int):
        return self._epsilon_greedy_policy(state)

    def update_episode_info(self, new_state, action, reward, t, episode_finished):
        # Crucial: Calculamos la probabilidad que tenía esa acción bajo la política exploratoria
        # antes de que la tabla Q cambie al final del episodio.
        pi_A = self._random_epsilon_greedy_policy(self.state) # Probabilities of choosing each action for this state
        action_probability = pi_A[action] # Probability of choosing the specified action

        self.episode.append((self.state, action, reward, action_probability))
        self.state = new_state

        if self.epsilon_decay:
            self.epsilon = min(1.0, 1000.0 / (t + 1))

    def update_post_episode(self):
        current_return = 0.0  # Terminal state (current return = 0)
        W = 1.0  # Peso del muestreo por importancia (W <- 1)

        for state, action, reward, behavior_prob in reversed(self.episode):
            current_return = reward + self.discount_factor * current_return # Current return equals to immediate reward + future return with a discount factor applied

            self.C[state, action] += W

            learning_rate = (W / self.C[state, action]) # Compute learning rate using W, the weight for this step

            expected_return = self.Q[state, action]
            self.Q[state, action] += learning_rate * (current_return - expected_return) # Update return table with the proper return for this step (bellman)

            # La política objetivo es greedy, devuelve la que mejor Q tenga
            best_action = np.argmax(self.Q[state])

            # REGLA CRÍTICA DE OFF-POLICY:
            # Si la acción que tomó nuestro agente exploratorio NO coincide con la acción óptima
            # que habría tomado la política codiciosa, la probabilidad de la política objetivo es 0.
            # Eso hace que W sea 0 para todos los pasos anteriores, por lo que dejamos de iterar.
            if action != best_action:
                break

            # Actualizamos el peso W para el siguiente paso hacia atrás:
            # W <- W * ( pi(A|S) / b(A|S) ). Como pi es greedy, pi(best_action|S) = 1.0
            W = W * (1.0 / behavior_prob)

        total_episode_reward = sum(reward for _, _, reward, _ in self.episode)
        self.episode = [] # Clean episode info for next episode

        return total_episode_reward

    def get_episode_len(self):
        return len(self.episode)

    def getQ(self):
        return self.Q

    def getCurrentEpsilon(self):
        return self.epsilon

