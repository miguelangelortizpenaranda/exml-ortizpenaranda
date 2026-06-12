# @title Funciones para mostrar los resultados
import math

import numpy as np
from matplotlib import pyplot as plt


def print_pi_star_from_q(env, Q):
  """
  Imprime la política óptima construida hasta ahora por el agente. Se utiliza a modo
  de represeentar la solución obtenida
  :param env: entorno sobre el que mostrar la política
  """
  done = False
  pi_star = np.zeros([env.observation_space.n, env.action_space.n])
  state, info = env.reset()  # start in top-left, = 0
  actions = ""
  while not done:
    action = np.argmax(Q[state, :])
    actions += f"{action}, "
    pi_star[state, action] = action
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

  print("Política óptima obtenida\n", pi_star, f"\n Acciones {actions} \n Para el siguiente grid\n", env.render())

def plot_agent_rewards(agents_stats, nrows=2, ncols=2, figsize=(12, 8)):
  """
  Muestra gráficamente la proporción de recompensa acumulada de un agente a lo largo de los episodios
  """
  fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
  axes = axes.flatten()

  for ax, agent in zip(axes, agents_stats.keys()):
    ax.plot(agents_stats[agent])
    ax.set_title(agent)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")

  plt.tight_layout()
  plt.show()


def plot_episode_lengths_for_agents(agents_episode_lengths, nrows=2, ncols=2,
                                    avg_window=500, figsize=(12, 8)):
  """
  Muestra gráficamente la longitud de los episodios, agregada en una ventana de tiempo de tamaño
  configurable, a lo largo del transcurso del entrenamiento.
  """
  fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
  axes = axes.flatten()

  for ax, agent in zip(axes, agents_episode_lengths.keys()):
    moving_avg = np.convolve(agents_episode_lengths[agent], np.ones(avg_window) / avg_window, mode='valid')
    avg_indexes = list(range(avg_window - 1, len(agents_episode_lengths[agent])))

    ax.plot(avg_indexes, moving_avg)
    ax.set_title(agent)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")

  plt.tight_layout()
  plt.show()


def print_max_reward_proportion_for_agents(agents_stats):
  """
  Imprime la máxima proporción de recompensa acumulada de un agente (la última que se obtuvo)
  """
  for agent in agents_stats.keys():
    print(f"[{agent}] Máxima proporcion recompensa: {agents_stats[agent][-1]}")


def print_max_correctly_finished_episodes_for_agents(agent_correctly_finished_episodes):
  """
  Imprime el número de episodios completados correctamente para un agente
  """
  for agent in agent_correctly_finished_episodes.keys():
    print(f"[{agent}] Episodios terminados correctamente: {agent_correctly_finished_episodes[agent]}")


def print_correct_finished_episodes_length_avg_for_agents(agent_avg_correct_episode_lengths):
  """
  Imprime el promedio de longitud por episodios completados correctamente para un agente
  """
  for agent in agent_avg_correct_episode_lengths.keys():
    print(f"[{agent}] Promedio de longitud por episodio completado correctamente: {agent_avg_correct_episode_lengths[agent]}")


def print_global_best_for_finished_episodes(results_correctly_finished_episodes):
  """
  Imprime, para un conjunto de agentes, el que mayor número de episodios ha terminado satisfactoriamente,
  cuál es ese valor, y con qué hiperparámetros lo consiguió
  """
  best_params = ()
  best_agent = "None"
  best_val = 0.0
  for param_key in results_correctly_finished_episodes.keys():
    for agent in results_correctly_finished_episodes[param_key].keys():
      finished = results_correctly_finished_episodes[param_key][agent]
      if finished != 0 and finished > best_val:
        best_val = finished
        best_agent = agent
        best_params = param_key
  print(f"El agente que mayor número de episodios ha terminado ha sido: {best_agent} con {best_val} episodios para los parámetros {best_params}")


def print_global_best_for_episodes_length_avg(results_avg_correct_episode_lengths):
  """
  Imprime, para un conjunto de agentes, el que menor promedio de longitud por episodio finalizado satisfactoriamente tiene,
  cuál es ese valor, y con qué hiperparámetros lo consiguió
  """
  best_params = ()
  best_agent = "None"
  best_val = math.inf
  for param_key in results_avg_correct_episode_lengths.keys():
    for agent in results_avg_correct_episode_lengths[param_key].keys():
      avg_length = results_avg_correct_episode_lengths[param_key][agent]
      if avg_length != 0 and avg_length < best_val:
        best_val = avg_length
        best_agent = agent
        best_params = param_key
  print(f"El agente que mas rápido ha completado los episodios ha sido: {best_agent} con {best_val} pasos en promedio para los parámetros {best_params}")