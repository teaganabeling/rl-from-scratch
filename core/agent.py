# agent.py
# Base Agent class with abstract select_action() and learn() methods.
from abc import ABC, abstractmethod

from core.memory import ReplayBuffer
from core.config import load_config
config = load_config()

class Agent(ABC):
    def __init__(self, action_space, state_space):
        self.action_space = action_space # e.g., number of actions
        self.state_space = state_space # e.g., shape of state
        self.replay_buffer = ReplayBuffer()

    @abstractmethod
    def select_action(self):
        """Choose an action given the current state."""
        pass

    @abstractmethod
    def learn(self):
        """Update the agent's parameters from experience."""
        pass
