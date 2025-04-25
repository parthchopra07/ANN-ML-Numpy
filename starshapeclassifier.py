import numpy as np
import matplotlib.pyplot as plt

# Define the activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define a function to generate a star-shaped classification boundary
def inside_star(x, y, num_vertices=5, outer_radius=30, inner_radius=15):
    """Check if point (x,y) is inside a star shape"""
    angle = np.arctan2(y, x) % (2 * np.pi)  # Normalize angle between 0 and 2*pi
    distance = np.sqrt(x**2 + y**2)
    sector = int(angle / (2 * np.pi / num_vertices))  # Determine which sector the point is in
    boundary_angle = (2 * np.pi / num_vertices) * sector + np.pi / num_vertices
    if sector % 2 == 0:  # Outer vertex
        boundary_distance = outer_radius / np.cos(angle - boundary_angle)
    else:  # Inner vertex
        boundary_distance = inner_radius / np.cos(angle - boundary_angle)
    return distance < boundary_distance
# Generate a grid of points
x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
xx, yy = np.meshgrid(x, y)
mask = np.zeros_like(xx, dtype=bool)

# Apply the inside_star function to each point in the grid
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        mask[i, j] = inside_star(xx[i, j], yy[i, j])

# Plot the star shape
plt.figure(figsize=(8, 8))
plt.scatter(xx[mask], yy[mask], s=1)
plt.xlim([-50, 50])
plt.ylim([-50, 50])
plt.title('Star Shape')
#plt.show()
# Generate points and labels
np.random.seed(0)
points = np.random.uniform(low=0, high=100, size=(2000, 2))
labels = np.array([1 if inside_star(point[0]-50, point[1]-50) else 0 for point in points])

# Define a function to create and train the ANN
def train_ann(hidden_size, points, labels, epochs=1000, lr=0.01):
    input_size, output_size = 2, 1
    hidden_weights = np.random.uniform(size=(input_size, hidden_size))
    hidden_bias = np.random.uniform(size=(hidden_size,))
    output_weights = np.random.uniform(size=(hidden_size, output_size))
    output_bias = np.random.uniform(size=(output_size,))
    # Training the ANN
    for epoch in range(epochs):
        # Forward propagation
        hidden_layer_input = np.dot(points, hidden_weights) + hidden_bias
        hidden_layer_activations = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_activations, output_weights) + output_bias
        predicted_output = sigmoid(output_layer_input)

        # Backward propagation
        error = labels.reshape(-1, 1) - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activations)
        # Updating weights and biases
        output_weights += hidden_layer_activations.T.dot(d_predicted_output) * lr
        output_bias += np.sum(d_predicted_output, axis=0) * lr
        hidden_weights += points.T.dot(d_hidden_layer) * lr
        hidden_bias += np.sum(d_hidden_layer, axis=0) * lr

    return hidden_weights, hidden_bias, output_weights, output_bias

# Define a function to plot decision boundaries
def plot_decision_boundary(weights_biases, points, labels, title):
    hidden_weights, hidden_bias, output_weights, output_bias = weights_biases
    x_min, x_max = 0, 100
    y_min, y_max = 0, 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Forward propagation to get predictions
    hidden_layer_input = np.dot(np.c_[xx.ravel(), yy.ravel()], hidden_weights) + hidden_bias
    hidden_layer_activations = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activations, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_input)
    Z = (predicted_output > 0.5).astype(int)

    # Reshape result back into a grid
    Z = Z.reshape(xx.shape)

    # Plot contour and training points
    plt.contourf(xx, yy, Z, alpha=0.8, levels=[0, 0.5, 1], cmap=plt.cm.Paired)
    plt.scatter(points[:, 0], points[:, 1], c=labels, s=20, edgecolor='k', cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()


# Sizes of hidden layers to experiment with
hidden_layer_sizes = [2, 6, 20]

# Training ANNs and plotting decision boundaries
for size in hidden_layer_sizes:
    # Train ANN
    weights_biases = train_ann(size, points, labels)

    # Plot decision boundary
    plot_decision_boundary(weights_biases, points, labels, f'Decision Boundary with {size} Hidden Neurons')
