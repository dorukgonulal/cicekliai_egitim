import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Features (e.g., hours studied)
y = 4 + 3 * X + np.random.randn(100, 1)  # Labels (e.g., test scores with noise)

# Step 2: Implement Linear Regression
# Add a column of ones to X for the intercept term
X_b = np.c_[np.ones((100, 1)), X]  # 100x2 matrix: first column ones, second column X

# Calculate optimal weights (theta) using the Normal Equation: theta = (X_b^T * X_b)^(-1) * X_b^T * y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Extract slope and intercept for the line
intercept, slope = theta_best[0, 0], theta_best[1, 0]
print(f"Intercept: {intercept:.2f}, Slope: {slope:.2f}")

# Step 3: Make predictions
X_new = np.array([[0], [2]])  # New data points (e.g., hours studied)
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add intercept column to new data
y_predict = X_new_b.dot(theta_best)  # Predicted values

# Step 4: Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.6, label="Training Data")  # Original data points
plt.plot(X_new, y_predict, color='red', linewidth=2, label="Prediction Line")  # Prediction line

plt.xlabel("Hours Studied")
plt.ylabel("Test Score")
plt.title("Linear Regression: Predicting Test Scores based on Study Hours")
plt.legend()
plt.show()
