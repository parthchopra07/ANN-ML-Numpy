import numpy as np

def logistic(z):
    return 1 / (1 + np.exp(-z))

def logistic_derivative(z):
    return logistic(z) * (1 - logistic(z))

v1 = np.array([0.0, 0.0])
v2 = np.array([0.0, 0.0])
w = np.array([0.0, 0.0])

X = np.array([
    [1, 3],
    [3, 1],
    [1, 0.5],
    [2, 2],
    [0.5, 0.5],
    [2, 3],
    [1, 1],
    [1, 0.5],
    [0.5, 2],
    [0.5, 1]
])

Y = np.array([1, 1, -1, -1, -1, 1, -1, -1, 1, -1])

epochs = 10000
lr = 0.01

for epoch in range(epochs):
    # Forward pass
    h1 = logistic(np.dot(X, v1))
    h2 = logistic(np.dot(X, v2))
    y_pred = logistic(w[0] * h1 + w[1] * h2)
    
    # Error calculation
    error = Y - y_pred
    
    # Backward pass - Gradient calculations
    # Gradient for the weights of the output layer
    d_w = np.dot(error, np.vstack((h1, h2)).T)
    
    # Gradients for the weights of the input layer
    d_v1 = np.dot((error * w[0] * logistic_derivative(np.dot(X, v1))).T, X).T
    d_v2 = np.dot((error * w[1] * logistic_derivative(np.dot(X, v2))).T, X).T
    
    # Update weights
    v1 += lr * d_v1
    v2 += lr * d_v2
    w += lr * d_w
    
    # Print out the loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(error ** 2)
        print(f"Epoch {epoch}, Loss: {loss}")

# Test predictions after training
test_points = np.array([
    [1, 3],
    [0.5, 1],
    [1, 1]
])

for test_point in test_points:
    h1_test = logistic(np.dot(test_point, v1))
    h2_test = logistic(np.dot(test_point, v2))
    y_test_pred = logistic(w[0] * h1_test + w[1] * h2_test)
    print(f"Point {test_point} prediction: {'No collision' if y_test_pred >= 0.5 else 'Collision'}")
