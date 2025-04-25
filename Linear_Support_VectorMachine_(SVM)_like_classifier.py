import numpy as np
def phi(x):
    return np.array([1, x[0], x[1], x[0]**2 + x[1]**2])
# Generate some random data points
np.random.seed(0)
data_points = np.random.randn(200, 2) * 2
transformed_data = np.array([phi(point) for point in
data_points])
radius = 2
labels = np.array([1 if point[0]**2 + point[1]**2 <=
radius**2 else -1 for point in data_points])
class HingeLossFitting:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, epochs=1000):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for _ in range(epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * np.dot(x_i, self.weights) >= 1
                if not condition:
                    # Update only for misclassified or within-margin points
                    dw = -np.dot(x_i, y[idx])
                    self.weights -= dw
                    
    def predict(self, X):
        return np.sign(np.dot(X, self.weights))
# Train the model without regularization
model = HingeLossFitting()
model.fit(transformed_data, labels, epochs=1000)