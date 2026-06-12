import numpy as np
from algorithms.algorithm import Algorithm

try:
    from algorithms.algorithm import Algorithm
except ImportError:
    from .algorithm import Algorithm

class Softmax(Algorithm):
    def __init__(self, k: int, temperature: float = 0.5):
        """
        Inicializa el algoritmo Softmax.

        :param k: Número de brazos.
        :param temperature: Parámetro tau (τ) que controla la exploración.
        :raises ValueError: Si la temperatura es <= 0.
        """
        assert temperature > 0, "El parámetro temperatura (tau) debe ser mayor que 0."
        
        super().__init__(k)
        self.temperature = temperature

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la distribución de probabilidad Softmax.

        :return: índice del brazo seleccionado.
        """
        # Calculamos la exponencial de los valores normalizados por la temperatura (tau)
        # Nota: Restamos el máximo de self.values por estabilidad numérica (evita overflow)
        exp_values = np.exp((self.values - np.max(self.values)) / self.temperature)
        
        # Calculamos las probabilidades siguiendo la función de Gibbs
        probabilities = exp_values / np.sum(exp_values)
        
        # Seleccionamos el brazo según la distribución calculada
        chosen_arm = np.random.choice(self.k, p=probabilities)
        
        return chosen_arm