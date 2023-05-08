import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.05, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def _standardize_features(self, features):
        means = np.mean(features, axis=0)
        std_devs = np.std(features, axis=0)
        normalized_features = (features - means) / std_devs
        return normalized_features, means, std_devs

    def _gradient_descent(self, features, target):
        samples, num_features = features.shape
        theta_values = np.zeros(num_features + 1)
        features = np.column_stack((np.ones(samples), features))

        for iteration in range(self.iterations):
            errors = np.dot(features, theta_values) - target
            gradients = (1 / samples) * np.dot(features.T, errors)
            theta_values -= self.learning_rate * gradients
        return theta_values

    def fit(self, features, target):
        self.features, self.data_mean, self.data_std = self._standardize_features(features)
        self.theta = self._gradient_descent(self.features, target)

# Load data from files
data_X = np.loadtxt('DataX.dat')
data_Y = np.loadtxt('DataY.dat')

# Train the model
linear_reg = LinearRegression(learning_rate=0.05, iterations=100)
linear_reg.fit(data_X, data_Y)

# Gradient descent function with visualization
def gradient_descent_visualization(features, target, theta, learning_rate, iterations):
    samples, num_features = features.shape
    features = np.column_stack((np.ones(samples), features))
    cost_history = []

    for iteration in range(iterations):
        errors = np.dot(features, theta) - target
        gradients = (1 / samples) * np.dot(features.T, errors)
        theta -= learning_rate * gradients
        cost = (1 / (2 * samples)) * np.sum(np.square(errors))
        cost_history.append(cost)

    # Plot cost after each iteration
    for i, cost in enumerate(cost_history):
        plt.plot(cost_history[:i + 1])
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost function after each iteration')
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()

    return theta, cost_history

theta, cost_history = gradient_descent_visualization(linear_reg.features, data_Y, linear_reg.theta, 0.05, 100)
