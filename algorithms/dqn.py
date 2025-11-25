import numpy as np
from core.agent import Agent
from core.neural import NeuralNetwork
from core.memory import ReplayBuffer

from core.config import load_config
config = load_config()

class DQN(Agent):
    def __init__(self, state_size, action_size):
        super().__init__(action_size, state_size)  # Initialize base Agent class

        self.q_network = NeuralNetwork(state_size, action_size, 'linear').model
        self.target_network = NeuralNetwork(state_size, action_size, 'linear').model
        self.replay_buffer = ReplayBuffer()  # Experience replay buffer
        self.learn_step = 0  # Counter for learning steps

        self.gamma = config['agent']['gamma']  # Discount factor
        self.batch_size = config['replay_buffer']['batch_size']  # Mini-batch size
        self.update_frequency = config['agent']['update_frequency']  # How often to update the target network
        self.epsilon = config["agent"]["epsilon_start"]  # Default epsilon value
        self.epsilon_end = config["agent"]["epsilon_end"]
        self.epsilon_decay = config["agent"]["epsilon_decay"]
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon: # If random number < epsilon, explore
            return np.random.randint(self.q_network.output_shape[-1]) # Take a random action: [-1] gets the size of the action space
        else: # Else, exploit
            return np.argmax(self.q_network.predict(np.array([state]), verbose=0)[0]) # Choose action with highest Q-value
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to learn
        
        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size) # Sample a batch from replay buffer
        
        # Predict Q-values for current states and next states
        target_q = self.q_network.predict(states, verbose=0)  # Predict Q-values for current states
        next_q = self.target_network.predict(next_states, verbose=0)  # Predict Q-values for next states, "verbose" controls how much output you see during training

        for i in range(self.batch_size): 
            target = rewards[i]  # Immediate reward
            if not dones[i]:  # If not terminal state
                target += self.gamma * np.amax(next_q[i])  # Add discounted max future reward
            target_q[i][actions[i]] = target  # Update the Q-value for the taken action
    
        history = self.q_network.fit(states, target_q, epochs=1, verbose=0)  # Train the Q-network
        loss = history.history['loss'][0]
        
        self.learn_step += 1
        if self.learn_step % self.update_frequency == 0:  # If time to update the network
            self.target_network.set_weights(self.q_network.get_weights())  # Update target network
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)  # Decay epsilon
        
        return loss
