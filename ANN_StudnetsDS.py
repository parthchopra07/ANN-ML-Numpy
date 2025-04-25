
'''
Sample ANN question for assignmnet 2(Behzad akbari)
Date 11 March 2024
In this code, I am synthesizing a dataset to predict the percentage of students who can pass an ANN course,
focusing on two features: attendance in class and dedication of 10 hours each week to the course for assignments,
 quizzes, and exams. We'll construct the dataset and split it into two parts: training and testing. Then, we'll
 use an ANN with two features and two layers to model the system and test whether an individual who studies for
 4 hours and attends 50% of the classes is likely to pass the course.
'''
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize parameters for the Neural Network
input_size = 2  # Number of features
hidden_layer_size = 5  # Number of nodes in the hidden layer
output_size = 1  # Number of output nodes
np.random.seed(42) # To have same set of random data for reproducibility

# Random weights and bias initialization
weights_input_to_hidden = np.random.uniform(-1, 1, (input_size, hidden_layer_size))# for instance our "v" and this will give uniform values from -1 to +1
weights_hidden_to_output = np.random.uniform(-1, 1, (hidden_layer_size, output_size))#for instance our "w"
bias_hidden = np.random.uniform(-1, 1, (1, hidden_layer_size))
bias_output = np.random.uniform(-1, 1, (1, output_size))
#The terms in the brackets (sizes) define the dimensions of the output matrices

# Learning rate and epochs
learning_rate = 0.01
epochs = 1000

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=input_size,n_redundant=0, n_clusters_per_class=1, n_classes=2)
y = y.reshape(-1, 1)  # Reshape y to be a column vector

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Training the Neural Network
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X_train, weights_input_to_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    final_output = sigmoid(final_output_layer_input)

    # Backward pass
    error = y_train - final_output

    d_final_output = error * sigmoid_derivative(final_output_layer_input)
    error_hidden_layer = d_final_output.dot(weights_hidden_to_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_input)

    # Update the weights and biases
    weights_hidden_to_output += hidden_layer_output.T.dot(d_final_output) * learning_rate
    bias_output += np.sum(d_final_output, axis=0, keepdims=True) * learning_rate

    weights_input_to_hidden += X_train.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 100 == 0:
        loss = np.mean(np.square(error))
        print(f'Epoch {epoch}, Loss: {loss}')
# After training, let's predict a single student example with 50% attendance and 4 hours of studying
student_data = np.array([[0.5, 4/10]])  # Example student data

# Forward pass for prediction
hidden_layer_input = np.dot(student_data, weights_input_to_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)
final_output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
student_prediction = sigmoid(final_output_layer_input)

# The prediction will be closer to 1 if the student is likely to pass, and closer to 0 if not
print(student_prediction)


