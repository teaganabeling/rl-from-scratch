from abc import ABC, abstractmethod

class Environment(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def reset(self):
        """Reset environment and return initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """Take action, return (state, reward, done, info)."""
        pass

    @abstractmethod
    def render(self):
        """Render environment (print or visualize)."""
        pass