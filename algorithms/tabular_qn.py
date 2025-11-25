# tabular_qn.py

import numpy as np
from core.agent import Agent
from envs.gridworld_env import GridworldEnv

from core.config import load_config
config = load_config()

class TabularQN(Agent):
    def __init__(self, nrow, ncol, n_actions):
        self.nrow = nrow
        self.ncol = ncol
        self.n_actions = n_actions
        
        # Initialize Q-table: shape (nrow, ncol, n_actions)
        self.Q = np.zeros((nrow, ncol, n_actions))

    def select_action(self, state, epsilon):
        x, y = state
        if np.random.rand() < epsilon: # If random number < epsilon, explore
            return np.random.randint(self.n_actions) # Take a random 
        else: # Else, exploit
            return np.argmax(self.Q[x, y, :]) # Take the action with highest Q-value
    
    def learn(self, state, action, reward, next_state, done):
        alpha = config['agent']['alpha']
        gamma = config['agent']['gamma']
        x, y = state
        nx, ny = next_state
        best_next = np.max(self.Q[nx, ny, :])
        self.Q[x, y, action] += alpha * (reward + gamma * best_next - self.Q[x, y, action])

    def save_q_table_as_csv(self, filename="q_table.csv"):
        import pandas as pd
        # Flatten the 3D Q-table to 2D for CSV (row*col x action)
        q_flat = self.Q.reshape(self.nrow * self.ncol, self.n_actions)
        df = pd.DataFrame(q_flat)
        df.to_csv(filename, index=False, header=False)
