import gymnasium as gym

class MountainCarRewardEnvironment(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)

        position = next_state[0]
        speed = next_state[1]

        shaped_reward = -1.0 + abs(position + 0.5) + (speed ** 2) * 10.0

        return next_state, shaped_reward, terminated, truncated, info