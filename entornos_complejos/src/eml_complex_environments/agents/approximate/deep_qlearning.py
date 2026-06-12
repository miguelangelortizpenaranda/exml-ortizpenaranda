import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .approximate_agent import ApproximateAgent

NN_FEATURES = 64

class QNetwork(nn.Module):
    """
    Red neuronal para aproximar la función Q.
    :param state_dimension: Dimensión del estado.
    :param action_dimension: Número de acciones posibles.
    :param hidden_dimension: Número de neuronas en las capas ocultas.
    """
    def __init__(self, state_dimension, action_dimension, hidden_dimension=NN_FEATURES):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dimension, hidden_dimension) # Capa de entrada (estado, continuo, en el que estamos), y 64 características (neuronas) de salida
        self.layer2 = nn.Linear(hidden_dimension, hidden_dimension) # Capa oculta de 64 neuronas a 64 neuronas
        self.layer3 = nn.Linear(hidden_dimension, action_dimension) # Capa de salida, produce un valor por cada acción posible

    def forward(self, x):
        """
        Propagación hacia adelante, de cara a realizar las predicciones.
        :param x: Estado de entrada con forma [samples_count, state_dimension]
        :return: Valores Q para cada acción, con forma [samples_count, action_dimension]
        """
        # Aplicar la primera capa de entrada seguida de ReLU.
        x = F.relu(self.layer1(x))
        # Aplicar la segunda capa oculta seguida de ReLU.
        x = F.relu(self.layer2(x))
        # Capa de salida sin activación, para obtener los valores Q.
        x = self.layer3(x)
        return x


class DeepQLearning(ApproximateAgent):
    """
    Implementa un agente que utiliza el método de Deep Q-Learning, donde existen
    dos redes neuronales, una que dictamina la política de comportamiento, y otra
    que define la política objetivo, que se van entrenando con batchs (conjuntos completos)
    de muestras a a lo largo de distintos episodios. Para esto, se utila un replay_buffer,
    de tamaño configurable, que nos permite recordar experiencias pasadas.

    Ha sido necesario para hacer funcionar a este agente introducir un wrapper del entorno original
    que modificara la forma en la que se aportan las recompensas, de forma que se premie cuando
    el coche está más cerca del objetivo, así como la propia inercia que lleva el mismo en aras
    de conseguir su cometido.
    """
    def __init__(self, env, epsilon, discount_factor, learning_rate, random_seed,
                 epsilon_decay=True, decay_factor=0.997, buffer_size=10000, batch_size=64, target_update_steps=1000):

        super().__init__(env, epsilon, random_seed)

        # Fijamos semillas aleatorias para reproducibilidad
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.decay_factor = decay_factor

        self.batch_size = batch_size # Número de experiencias que la red neuronal procesa a la vez para actualizar sus pesos
        self.target_update_frequency = target_update_steps # Número de pasos en los que se actualiza la red que fija la política objetivo
        self.total_steps = 0 # Pasos totales transcurridos

        state_dim = self.env.observation_space.shape[0] # Estado continuo de observación (posición, velocidad)

        # Red neuronal de comportamiento (behaviour), utilizada para elegir las acciones a realizar
        self.q_network = QNetwork(state_dimension=state_dim, action_dimension=self.n_actions, hidden_dimension=NN_FEATURES)

        # Red neuronal usada para la política objetivo. Solo se actualiza cada cierta cantidad de pasos, y se utiliza
        # para calcular el retorno esperado del estado siguiente, de forma que quede congelado unos pasos
        self.target_network = QNetwork(state_dimension=state_dim, action_dimension=self.n_actions, hidden_dimension=NN_FEATURES)

        # Queremos que los pesos de la target_network sean inicialmente similares a los de
        # la q_network, pues inicialmente se generan con pesos aleatorios
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Se encarga de ir modificando y optimizando los pesos de la red, usando descenso de gradiente
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Función de pérdida. En este caso se usa el error cuadrático medio
        # Se calcula la dirección del error (el gradiente), de cara a ajustar los pesos
        self.loss_function = nn.MSELoss()

        # Se utiliza un replay buffer para guardar una memoria que permita al agente
        # ser entrenado con recuerdos pasados
        self.replay_buffer = deque(maxlen=buffer_size)

        self.current_state = None
        self.episode_return = 0.0
        self.episode_len = 0

    def set_initial_state(self, state):
        """
        Fija el estado inicial para el agente
        :param state: estado inicial procedente del entorno
        """
        self.current_state = state
        self.episode_len = 0

    def get_action(self, state):
        """
        Devuelve una acción a tomar por el agente, dependiendo del estado donde se encuentre
        :param state: estado actual
        :return: accion a tomar por el agente
        """
        return self._epsilon_greedy_policy(state)

    def on_episode_step(self, new_state, action, reward, t, episode_finished=False):
        """
        Llamado cuando un nuevo paso ha transcurrido en el episodio actual. En caso
        de que se tengan suficientes muestras como para reentrenar la red:
        - Se elige un conjunto de muestras aleatorias de nuestro histórico
        - Se obtiene el retorno actual usando la red de la política de comportamiento para esas muestras
        - Se obtiene el retorno esperado usando la política objetivo
        - Se calcula el error (RMSE) usando esos dos valores, y se reentrena la política de comportamiento
        - Cada cierto numero de pasos, los pesos de esa red se copian y quedan congelados en la red objetivo

        :param new_state: nuevo estado tras tomar la acción especificada
        :param action: acción tomada por el agente
        :param reward: recompensa obtenida tras tomar esa acción
        :param t: número de episodio actual
        :param episode_finished: si el episodio ha sido completado
        """
        # Se guarda en la memoria el estado actual y acción tomada, junto con la recompensa
        # (que recordemos que se modifica en un wrapper), y si el episodio ha terminado
        self.replay_buffer.append((self.current_state, action, reward, new_state, episode_finished))

        # Entrenamos la red si ya tenemos suficientes muestras (batch_size)
        if len(self.replay_buffer) >= self.batch_size:

            batch = random.sample(self.replay_buffer, self.batch_size) # Se cogen tantas muestras aleatorias como batch_size
            states, actions, rewards, new_states, dones = zip(*batch) # Se separan en todas sus componentess

            states_tensor = torch.FloatTensor(np.array(states)).view(self.batch_size, -1) # Transforma a matriz de batch_size x columnas de estados actuales
            new_states_tensor = torch.FloatTensor(np.array(new_states)).view(self.batch_size, -1) # Transforma a matriz de batch_size x columnas de estados siguientes

            actions_tensor = torch.LongTensor(actions).unsqueeze(1) # Matriz que asocia acción elegida por cada elemento del batch
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1) # Matriz que asocia recompensa obtenida por cada elemento del batch
            dones_tensor = torch.FloatTensor(dones).unsqueeze(1) # Matriz que asocia si se ha terminado el episodio por cada elemento del batch
            # Convertimos si se ha terminado a 1 y si no se ha terminado a 0 para poder operar más fácil

            current_return = self.q_network(states_tensor).gather(1, actions_tensor) # Realiza la predicción del valor q actual
            # en base a la acción que se tomó para cada muestra del batch (la coteja con cada elemento de actions_tensor)

            # Predicción Q objetivo
            # Desactivamos el cálculo de gradientes (no es necesario en modo evaluación)
            with torch.no_grad():
                q_values = self.target_network(new_states_tensor)
                max_expected_return = q_values.max(1)[0].unsqueeze(1) # Seleccionar máximo q de los nuevos estados
                # Igual que en QLearning tradicional, si se ha terminado el episodio, el retorno es la recompensa,
                # en caso contrario, es el retorno futuro con un factor de descuento aplicado
                target_return = rewards_tensor + (1 - dones_tensor) * self.discount_factor * max_expected_return

            # Se calcula el error usando RMSE, y el optimizador, para ir modificando
            # los pesos de la red q_network en base al mismo
            loss_f = self.loss_function(current_return, target_return)
            self.optimizer.zero_grad() # Pone todos los gradientes a none
            loss_f.backward() # Backpropagation, donde se calcula el gradiente de cada peso para ver que neuronas son las causantes del error
            self.optimizer.step() # Se modifican los pesos de la red en base al paso anterior

        self.current_state = new_state
        self.episode_return -= 1 # Como hemos usado un wrapper para la recompensa, necesitamos poner el reward original a mano
        self.episode_len += 1
        self.total_steps += 1

        # Solo actualiza la red objetivo cada target_update_frequency pasos
        if self.total_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


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
        Obtiene la red neuronal que contiene los valores q de este agente
        """
        return self.q_network

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
        return self.env

    def _epsilon_greedy_policy(self, state):
        """
        Política epsilon-greedy, que, con una probabilidad epsilon,
        devuelve una acción aleatoria, y con una probabilidad de 1-epsilon
        devuelve la mejor acción dado el estado actual y la política
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Obtiene la matriz del estado
        state_tensor = torch.FloatTensor(state).view(1, -1)
        # Para un estado, calcula obtiene el valor para todas las acciones
        # de la red, y selecciona aquella con el máximo valor q
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()
