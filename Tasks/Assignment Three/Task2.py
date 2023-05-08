import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.05, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def _normalize_features(self, features):
        feature_mean = np.mean(features, axis=0)
        feature_std = np.std(features, axis=0)
        normalized_features = (features - feature_mean) / feature_std
        return normalized_features, feature_mean, feature_std

    def _logistic_function(self, input_val):
        return 1 / (1 + np.exp(-input_val))

    def _cost_function(self, theta, features, target):
        num_samples = len(target)
        predictions = self._logistic_function(np.dot(features, theta))
        cost = -(1 / num_samples) * (np.dot(target, np.log(predictions)) + np.dot((1 - target), np.log(1 - predictions)))
        return cost

    def _gradient_descent(self, features, target):
        num_samples, num_features = features.shape
        theta = np.zeros(num_features + 1)
        features = np.column_stack((np.ones(num_samples), features))
        cost_history = []

        for i in range(self.iterations):
            predictions = self._logistic_function(np.dot(features, theta))
            gradient = (1 / num_samples) * np.dot(features.T, predictions - target)
            theta -= self.learning_rate * gradient
            cost_history.append(self._cost_function(theta, features, target))

        return theta, cost_history

    def fit(self, features, target):
        self.features, self.data_mean, self.data_std = self._normalize_features(features)
        self.theta, self.cost_history = self._gradient_descent(self.features, target)

    def plot_functions(self):
        input_range = np.linspace(-10, 10, 100)
        threshold_func = np.where(input_range > 0, 1, 0)
        logistic_func = self._logistic_function(input_range)

        plt.plot(input_range, threshold_func, label='Threshold Function')
        plt.plot(input_range, logistic_func, label='Logistic Function')
        plt.xlabel('z')
        plt.ylabel('g(z)')
        plt.legend()
        plt.title('Threshold and Logistic Functions')
        plt.show()

    def plot_data_points(self, features, target):
        plt.scatter(features[target == 1][:, 0], features[target == 1][:, 1], c='b', marker='+', label='Suitable')
        plt.scatter(features[target == 0][:, 0], features[target == 0][:, 1], c='y', marker='o', label='Not Suitable')
        plt.xlabel('Living Area')
        plt.ylabel('Number of Bedrooms')
        plt.legend()

    def plot_decision_boundary(self, features, target):
        self.plot_data_points(features, target)
        X1_min, X1_max = features[:, 0].min(), features[:, 0].max()
        X2_min, X2_max = features[:, 1].min(), features[:, 1].max()
        xx1, xx2 = np.meshgrid(np.linspace(X1_min, X1_max), np.linspace(X2_min, X2_max))
        grid = np.column_stack((np.ones_like(xx1.ravel()), xx1.ravel(), xx2.ravel()))
        h = self._logistic_function(np.dot(grid, self.theta)).reshape(xx1.shape)
        plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
        plt.title('Decision Boundary')
        plt.show()

# Load data from files
data_X = np.loadtxt('DataX.dat')
data_Y = np.loadtxt('ClassY.dat')

# Train the model
log_reg = LogisticRegression(learning_rate=0.05, iterations=1000)
log_reg.fit(data_X, data_Y)

# Plot threshold and logistic functions
log_reg.plot_functions()

# Visualize data and decision boundary
log_reg.plot_decision_boundary(data_X, data_Y)

