import gymnasium as gym

class MountainCarRewardEnvironment(gym.Wrapper):
    """
    Implementa una clase que hereda de Wrapper, que nos permite modificar el método
    step para alterar la recompensa que se otorga, de forma que no solo tenga en cuenta
    el descuento por tiempo, sino la posición con respecto al objetivo, así como la velocidad
    del coche.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)

        position = next_state[0]
        speed = next_state[1]
        # Modificamos la recompensa para tener en cuenta el descuento por tiempo, pero también la posición
        # con respecto al objetivo, y la velocidad
        shaped_reward = -1.0 + abs(position + 0.5) + (speed ** 2) * 10.0

        return next_state, shaped_reward, terminated, truncated, info