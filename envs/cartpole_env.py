import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from envs.environment import Environment

from core.config import load_config
from core.utils import set_seed

class CartpoleEnv(Environment):

    def __init__(self, config):
        super().__init__(config) # Class the constuctor of the parent class (Environment) and passes the config object to it
        
        max_steps = config["training"]["max_steps"]

        render_mode = config["cartpole"].get("render_mode", "rgb_array")
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.env = TimeLimit(self.env, max_episode_steps=max_steps) # Wrap environment with a time limit

        self.action_space = self.env.action_space 
        self.state_space = self.env.observation_space

    def reset(self):
        state, info = self.env.reset(seed=self.config["env"]["seed"])
        return state, info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action) # Take a step in the environment using the provided action
        done = terminated or truncated  # "done" is True if the episode is terminated or truncated (Gym API convention)
        return state, reward, done, info

    def render(self):
        self.env.render() # Render the current state of the environment

    def close(self):
        self.env.close()
