from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def get_action(self, state: int):
        pass

    @abstractmethod
    def update_episode_info(self, new_state, action, reward):
        pass

    @abstractmethod
    def update(self):
        pass