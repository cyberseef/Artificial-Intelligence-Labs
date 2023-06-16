import numpy as np

# Define the activation function (e.g., sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the activation function (for backpropagation)
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.zeros((1, self.output_size))
    
    def forward(self, X):
        # Calculate the output of the neural network
        self.hidden_output = sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.hidden_output, self.weights2) + self.bias2)
        
        return self.output
    
    def backward(self, X, y, learning_rate):
        # Perform backpropagation to update weights and biases
        
        # Calculate the gradient of the loss with respect to the output
        d_output = (self.output - y) * sigmoid_derivative(np.dot(self.hidden_output, self.weights2) + self.bias2)
        
        # Calculate the gradient of the loss with respect to the hidden layer
        d_hidden = np.dot(d_output, self.weights2.T) * sigmoid_derivative(np.dot(X, self.weights1) + self.bias1)
        
        # Update weights and biases
        self.weights2 -= learning_rate * np.dot(self.hidden_output.T, d_output)
        self.bias2 -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.weights1 -= learning_rate * np.dot(X.T, d_hidden)
        self.bias1 -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)
    
    def train(self, X, y, epochs, learning_rate):
        # Train the neural network
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            # Calculate and print the loss (e.g., mean squared error)
            loss = np.mean((output - y) ** 2)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
