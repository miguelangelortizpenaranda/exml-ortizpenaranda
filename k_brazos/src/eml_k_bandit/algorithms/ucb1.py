
import math

import numpy as np
try:
    from algorithms.algorithm import Algorithm
except ImportError:
    from .algorithm import Algorithm

class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 1):
        """
        Inicializa el algoritmo UCB-1.

        :param k: Número de brazos.
        :param c: Parámetro de ajuste que controla el grado de exploración
        """
        super().__init__(k)
        self.c = c

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB-1.

        :return: índice del brazo seleccionado.
        """

        zeros = np.flatnonzero(self.counts == 0)
        # No han sido explorados todos los brazos
        if zeros.size > 0:
            chosen_arm = zeros[0]
            return chosen_arm

        total_counts = self.counts.sum()

        arms_ucbs = self.values + self.c * np.sqrt( 2 * np.log(total_counts) / self.counts ) # Numpy permite operaciones vectorizadas
        chosen_arm: int = int(np.argmax(arms_ucbs))
        return chosen_arm




