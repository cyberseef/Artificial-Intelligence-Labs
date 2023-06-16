import numpy as np

# Define the K-means clustering class
class KMeans:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.centroids = None
    
    def initialize_centroids(self, X):
        # Randomly initialize the centroids from the data points
        indices = np.random.choice(len(X), size=self.num_clusters, replace=False)
        self.centroids = X[indices]
    
    def assign_clusters(self, X):
        # Assign each data point to the closest centroid
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        cluster_labels = np.argmin(distances, axis=0)
        
        return cluster_labels
    
    def update_centroids(self, X, cluster_labels):
        # Update the centroids based on the mean of the data points in each cluster
        for i in range(self.num_clusters):
            self.centroids[i] = np.mean(X[cluster_labels == i], axis=0)
    
    def train(self, X, epochs):
        # Train the K-means clustering model
        
        # Initialize centroids
        self.initialize_centroids(X)
        
        for epoch in range(epochs):
            # Assign clusters
            cluster_labels = self.assign_clusters(X)
            
            # Update centroids
            self.update_centroids(X, cluster_labels)
            
            # Print the current centroids for each epoch
            print(f"Epoch {epoch+1}/{epochs}, Centroids: {self.centroids}")
