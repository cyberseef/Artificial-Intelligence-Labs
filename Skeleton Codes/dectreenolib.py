import numpy as np

# Define the Decision Tree node class
class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of the feature to split on
        self.threshold = threshold      # Threshold value for splitting
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Predicted value at the leaf node

# Define the Decision Tree class
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
    
    def find_best_split(self, X, y):
        # Find the best feature and threshold for splitting the data
        best_score = -np.inf
        best_feature_idx = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_idx])
            for threshold in unique_values:
                score = self.calculate_score(X, y, feature_idx, threshold)
                if score > best_score:
                    best_score = score
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold
    
    def calculate_score(self, X, y, feature_idx, threshold):
        # Calculate the score for a particular split (e.g., Gini Index, Information Gain)
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        
        left_y = y[left_mask]
        right_y = y[right_mask]
        
        # Calculate the impurity or loss function for the left and right subsets
        left_score = self.calculate_impurity(left_y)
        right_score = self.calculate_impurity(right_y)
        
        # Calculate the weighted average of the impurities or loss function
        score = (len(left_y) * left_score + len(right_y) * right_score) / len(y)
        
        return score
    
    def calculate_impurity(self, y):
        # Calculate the impurity or loss function (e.g., Gini Index, Entropy)
        # Implementation depends on the specific impurity measure
        
        # Example: Gini Index
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        impurity = 1 - np.sum(probabilities ** 2)
        
        return impurity
    
    def build_tree(self, X, y, depth):
        # Recursively build the decision tree
        
        # Base case: check if stopping criteria met
        if depth == self.max_depth or np.unique(y).size == 1:
            value = np.argmax(np.bincount(y))
            return DecisionTreeNode(value=value)
        
        # Find the best feature and threshold for splitting
        feature_idx, threshold = self.find_best_split(X, y)
        
        # Base case: check if no further split is possible
        if feature_idx is None or threshold is None:
            value = np.argmax(np.bincount(y))
            return DecisionTreeNode(value=value)
        
        # Split the data based on the best feature and threshold
        mask = X[:, feature_idx] <= threshold
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]
        
        # Recursively build the left and right subtrees
        left = self.build_tree(X_left, y_left, depth + 1)
        right = self.build_tree(X_right, y_right, depth + 1)
        
        # Create the decision tree node
        return DecisionTreeNode(feature_idx=feature_idx, threshold=threshold, left=left, right=right)
    
    def train(self, X, y):
        # Train the Decision Tree model
        
        # Build the decision tree
        self.root = self.build_tree(X, y, 0)
    
    def predict(self, X):
        # Predict the target values for new data points
        
        # Traverse the decision tree to make predictions
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            node = self.root
            while node.feature_idx is not None:
                if x[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions[i] = node.value
        
        return predictions
