import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Create MLP

class NeuralNetwork:
    def __init__(self, state_size, action_size, output_activation):
        self.state_size = state_size  # Number of input features (state dimensions)
        self.action_size = action_size  # Number of possible actions (output dimensions)
        self.model = self._build_model(output_activation)  # Build the neural network model

    def _build_model(self, output_activation):
        model = Sequential() # Initialize the neural network model

        model.add(Dense(24, input_shape=(self.state_size,), activation='relu')) # First hidden layer, input shape is the state size (24 neurons)
        model.add(Dense(24, activation='relu')) # Second hidden layer, also with 24 neurons, purpose is to learn complex features, patterns from input data
        model.add(Dense(self.action_size, activation=output_activation)) # Output layer, "linear" means we want to predict Q-values, one for each action
        model.compile(optimizer='adam', loss='mse') # Compile the model with Adam optimizer and Mean Squared Error loss function
        return model
    