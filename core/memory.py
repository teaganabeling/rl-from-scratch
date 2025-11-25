import numpy as np
from core.config import load_config
config = load_config()

class ReplayBuffer:
    def __init__(self):
        self.capacity = config["replay_buffer"]["capacity"] # Maximum number of experiences the buffer can hold
        self.data = []
        self.position = 0
    
    def __len__(self):
        return len(self.data)

    def store(self, state, action, reward, next_state, done): # Store a new experience
        if len(self.data) < self.capacity: # Add new experience if buffer not full
            self.data.append((state, action, reward, next_state, done)) # Append new experience
        else: # Overwrite oldest experience if buffer full
            self.data[self.position] = (state, action, reward, next_state, done) # Overwrite at current position
        self.position = (self.position + 1) % self.capacity # Update position circularly

    def sample_batch(self, batch_size): # Sample a batch of experiences
        batch_size = min(batch_size, len(self.data)) # Ensure batch size does not exceed current buffer size
        indices = np.random.choice(len(self.data), batch_size, replace=False) # Randomly select indices
        batch = [self.data[index] for index in indices] # Gather experiences
        states, actions, rewards, next_states, dones = zip(*batch) # Unzip experiences into separate arrays
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones) # Return as numpy arrays