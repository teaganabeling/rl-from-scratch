import numpy as np
from core.agent import Agent
from core.neural import NeuralNetwork
import tensorflow as tf

from core.config import load_config
config = load_config()

class A2C(Agent):

    def __init__(self, state_size, action_size):
        super().__init__(action_size, state_size)  # Initialize base Agent class
        self.state_size = state_size
        self.action_size = action_size

        self.q_network = NeuralNetwork(state_size, action_size, 'softmax').model # Policy network
        self.v_network = NeuralNetwork(state_size, 1, 'linear').model # State-value network

        self.learn_step = 0  # Counter for learning steps

        self.gamma = config['agent']['gamma']  # Discount factor
        self.learning_rate = config['agent']['learning_rate']  # Learning rate
        self.batch_size = config['replay_buffer']['batch_size']  # Mini-batch
        self.entropy_beta = config['agent']['entropy_beta']  # Entropy regularization coefficient

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.states, self.actions, self.rewards, self.dones, self.log_probs = [], [], [], [], []

    def select_action(self, state):
        state_flat = np.array(state).squeeze()
        self.states.append(state_flat)  # Store only the flat state
        state_batch = np.array([state_flat])  # Add batch dimension for prediction
        probs = self.q_network.predict(state_batch, verbose=0)[0]
        action = np.random.choice(self.action_size, p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def learn(self):
        if len(self.states) == 0:
            return None
        states = np.vstack(self.states).astype(np.float32)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        old_log_probs = np.array(self.log_probs)

        # Compute returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        values = self.v_network.predict(states, verbose=0).flatten()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        with tf.GradientTape() as tape_pi:
            probs = self.q_network(states, training=True)
            new_log_probs = tf.math.log(
                tf.reduce_sum(probs * tf.one_hot(actions, self.action_size), axis=1) + 1e-10
            )
            policy_loss = -tf.reduce_mean(new_log_probs * advantages)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))
            policy_loss -= self.entropy_beta * entropy  # Entropy regularization


            grads_pi = tape_pi.gradient(policy_loss, self.q_network.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(grads_pi, self.q_network.trainable_variables))

            # ---- Value update ----
            with tf.GradientTape() as tape_v:
                v_preds = tf.squeeze(self.v_network(states, training=True))
                value_loss = tf.reduce_mean((returns - v_preds) ** 2)
            grads_v = tape_v.gradient(value_loss, self.v_network.trainable_variables)
            self.value_optimizer.apply_gradients(zip(grads_v, self.v_network.trainable_variables))

        # Clear buffers
        self.states, self.actions, self.rewards, self.dones, self.log_probs = [], [], [], [], []

        loss = float(policy_loss + (0.5 * value_loss))
        return loss
