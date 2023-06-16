import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset or prepare your own data
X = ...  # Features
y = ...  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = np.mean((y_pred - y_test) ** 2)
r_squared = model.score(X_test, y_test)

# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("R^2 Score:", r_squared)
