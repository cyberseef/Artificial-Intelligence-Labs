import numpy as np
from sklearn.cluster import KMeans

# Load the dataset or prepare your own data
X = ...  # Features

# Create a K-means clustering model
model = KMeans(n_clusters=3)

# Fit the model to the data
model.fit(X)

# Get the cluster labels for each data point
labels = model.labels_

# Get the cluster centers
centers = model.cluster_centers_

# Perform clustering on new data
new_data = ...  # New data to be clustered
new_labels = model.predict(new_data)

# Access the inertia (sum of squared distances of samples to their closest cluster center)
inertia = model.inertia_

# Print the cluster labels, centers, and inertia
print("Cluster Labels:", labels)
print("Cluster Centers:", centers)
print("Inertia:", inertia)
