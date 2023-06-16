import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the logistic regression class
class LogisticRegression:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.random.randn(self.input_size, 1)
        self.bias = np.zeros((1, 1))
    
    def forward(self, X):
        # Calculate the output of logistic regression
        linear_output = np.dot(X, self.weights) + self.bias
        output = sigmoid(linear_output)
        
        return output
    
    def backward(self, X, y, learning_rate):
        # Perform gradient descent to update weights and bias
        
        # Calculate the gradient of the loss with respect to the weights and bias
        predictions = self.forward(X)
        d_weights = np.dot(X.T, predictions - y)
        d_bias = np.sum(predictions - y)
        
        # Update weights and bias
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
    
    def train(self, X, y, epochs, learning_rate):
        # Train the logistic regression model
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            # Calculate and print the loss (e.g., binary cross-entropy)
            loss = -(y * np.log(output) + (1 - y) * np.log(1 - output)).mean()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
