import numpy as np

from arms import Arm

class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param n: Recompensa máxima para esta distribución.
        :param p: Probabilidad de recompensa.
        """
        assert n > 0, "N debe ser un entero positivo."
        assert 0 <= p <= 1, "La probabilidad debe estar entre 0 y 1."

        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución binomial.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Genera una recompensa siguiendo una distribución binomial.

        :return: Recompensa obtenida del brazo.
        """
        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n: int = 10, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos que pueden otorgar una recompensa hasta n, con una probabilidad de entre p_min y p_max.

        :param k: Número de brazos a generar.
        :param n: Valor máximo de recompensa
        :param p_min: Probabilidad mínima de recompensa.
        :param p_max: Probabilidad máxima de recompensa.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0 <= p_min < p_max <= 1, "Rango de p inválido. Debe estar entre 0 y 1."

        p_vals = set()
        while len(p_vals) < k:
            p = np.random.uniform(p_min, p_max) # Generar k valores únicos de p
            p = round(p, 2)
            p_vals.add(p)

        p_vals = list(p_vals)

        arms = [ArmBinomial(n, p) for p in p_vals]

        return arms