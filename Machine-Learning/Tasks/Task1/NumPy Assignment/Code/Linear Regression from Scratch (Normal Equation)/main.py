import numpy as np

#We want to predict house prices based on house size (in square meters)
#We have the following data:
# House size (in square meters)
X = [50, 60, 80, 100, 120]
# House price (in thousands of dollars)
y = [150, 180, 240, 300, 330]

#Convert X and y into NumPy arrays
X = np.array(X)
print("X:\n", X)
y = np.array(y)
print("y:\n", y)
print("="*50)
# Reshape X into a column vector.
X_reshaped = X.reshape(-1, 1)
print("X_reshaped:\n", X_reshaped)
print("="*50)

#Add a bias column of ones to X
X_with_bias = np.hstack([np.ones((X_reshaped.shape[0], 1)), X_reshaped])
print("X_with_bias:\n", X_with_bias)
print("="*50)

#Compute the parameter vector θ using the Normal Equation: θ = (XT X)−1 XT y
X_transpose = X_with_bias.T
theta = np.linalg.inv(X_transpose @ X_with_bias) @ X_transpose @ y
print("theta:\n", theta)
print("="*50)

#Predict the price of a 90 square meter house
house_size = 90
house_size_reshaped = np.array([[house_size]])
house_size_reshaped_with_bias = np.hstack([np.ones((house_size_reshaped.shape[0], 1)), house_size_reshaped])
predicted_price = house_size_reshaped_with_bias @ theta
print("The predicted price of a 90 square meter house is:", predicted_price[0])
print("="*50)