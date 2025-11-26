# sarsa.py

import numpy as np
from core.agent import Agent
from envs.gridworld_env import GridworldEnv

from core.config import load_config
config = load_config()

class SARSA(Agent):
    def __init__(self, nrow, ncol, n_actions):
        self.nrow = nrow
        self.ncol = ncol
        self.n_actions = n_actions

        self.next_action = None
        self.Q = np.zeros((nrow, ncol, n_actions))

    def select_action(self, state, epsilon):
        x, y = state
        if np.random.rand() < epsilon: # If random number < epsilon, explore
            action = np.random.randint(self.n_actions) # Take a random 
        else: 
            action = np.argmax(self.Q[x, y, :]) # Take the action with highest Q-value
        self.next_action = action
        return action
    
    def learn(self, state, action, reward, next_state, done):
        alpha = config['agent']['alpha']
        gamma = config['agent']['gamma']
        x, y = state
        nx, ny = next_state

        target = reward
        if self.next_action is not None:
            target += gamma * self.Q[nx, ny, self.next_action] # r + gamma * Q(s'a')

        # Q(s,a) <-- Q(s,a) + alpha * [r + gamma * Q(s'a') - Q(s,a)]
        self.Q[x, y, action] += alpha * (target - self.Q[x, y, action])
    
    # TODO: Make modular
    def save_q_table_as_csv(self, filename="q_table.csv"):
        import pandas as pd
        # Flatten the 3D Q-table to 2D for CSV (row * col x action)
        q_flat = self.Q.reshape(self.nrow * self.ncol, self.n_actions)
        df = pd.DataFrame(q_flat)
        df.to_csv(filename, index=False, header=False)