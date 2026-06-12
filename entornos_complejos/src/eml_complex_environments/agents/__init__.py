# Importación de módulos o clases
from .tabular.monte_carlo_on_policy import MonteCarloOnPolicy
from .tabular.monte_carlo_off_policy import MonteCarloOffPolicy
from .tabular.qlearning import QLearning
from .tabular.sarsa import SARSA
from .approximate.sarsa_semi_gradient import SARSASemiGradient
from .approximate.deep_qlearning import DeepQLearning

from .agent import Agent

# Lista de módulos o clases públicas
__all__ = ['Agent','MonteCarloOnPolicy', 'MonteCarloOffPolicy', 'QLearning', 'SARSA', 'SARSASemiGradient', 'DeepQLearning']

