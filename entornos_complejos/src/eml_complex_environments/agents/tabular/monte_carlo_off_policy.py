import numpy as np

from .tabular_agent import TabularAgent

class MonteCarloOffPolicy(TabularAgent):
    """
    Implementación del agente Monte Carlo Off-Policy. El agente tiene en cuenta dos políticas, la política
    objetivo que es la que identifica las mejores acciones en cada caso, y es la que se aprende de la experiencia,
    y la política de comportamiento, que es la que interactúa con el entorno. El agente calcula sus retornos
    cuando llega al final de un episodio.
    """
    def __init__(self, environment, epsilon, discount_factor, random_seed, epsilon_decay):
        super().__init__(environment, epsilon, random_seed)

        self.C = np.zeros([self.n_states, self.n_actions]) # Matriz de importancias

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
        # Calculamos la probabilidad que tenía esa acción de ser elegida y la almacenamos en la información del episodio
        pi_A = self._random_epsilon_greedy_policy(self.state)
        action_probability = pi_A[action]
        self.episode.append((self.state, action, reward, action_probability))
        self.state = new_state

        # Calculamos epsilon decreciente si se ha configurado como tal
        if self.epsilon_decay:
            self.epsilon = min(1.0, self.decay_factor / (t + 1))

    def on_episode_finished(self):
        """
        Llamado cuando un episodio ha sido completado. En este caso, como tenemos
        la totalidad del histórico del episodio, lo recorremos hacia atrás para calcular
        los retornos esperados en la tabla Q(s, a), y actualizamos la política objetivo
        """
        current_return = 0.0  # Estado terminal, el retorno es 0
        weight = 1.0  # Peso para cada paso

        for state, action, reward, action_probability in reversed(self.episode):
            current_return = reward + self.discount_factor * current_return # El retorno actual equivale a la recompensa + returno futuro con factor de descuento aplicado

            self.C[state, action] += weight # Se incrementa la importancia en el peso

            learning_rate = (weight / self.C[state, action]) # Calcula la tasa de aprendizaje en función del peso para este paso

            expected_return = self.Q[state, action] # Obtenemos retorno esperado

            self.Q[state, action] += learning_rate * (current_return - expected_return) # Actualizamos la tabla de retornos con el valor para este estado-acción

            # La política objetivo va a ser greedy, devuelve la acción que mejor valor de Q tenga
            best_action = np.argmax(self.Q[state])

            # Si en algún punto del episodio se toma una acción distinta a la que tomaría la política de comportamiento,
            # ese episodio deja de ser válido para esa política objetivo,
            # con lo que lo anterior ya no aporta información útil
            if action != best_action:
                break

            # Actualizamos el peso para el siguiente paso hacia atrás, teniendo
            # en cuenta la probabilidad de que saliera esa acción
            weight =  weight / action_probability

        total_episode_reward = sum(reward for _, _, reward, _ in self.episode)
        self.episode = [] # Limpia la info del episodio

        return total_episode_reward


