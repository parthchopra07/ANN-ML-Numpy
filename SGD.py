import numpy as np
true_w = np.array([1, 2, 3, 4, 5])
d = len(true_w)
points = []
for i in range(100000): #Making Training Data
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn()
    points.append((x, y))

def sF(w, i): #Square Loss
    x, y = points[i]
    return (w.dot(x) - y)**2
def sdF(w, i): #Drivitive of Loss
    x, y = points[i]
    return 2*(w.dot(x) - y) * x    

def stochasticGradientDescent(sF, sdF, d, n):
    # Gradient descent
    w = np.zeros(d)
    numUpdates = 0
    for t in range(1000):
        for i in range(n):
            value = sF(w, i)
            gradient = sdF(w, i)
            numUpdates += 1
            eta = 1.0 / numUpdates
            w = w - eta * gradient
        print('iteration {}: w = {}, F(w) = {}'.format(t,w, value))
stochasticGradientDescent(sF, sdF, d,len(points))