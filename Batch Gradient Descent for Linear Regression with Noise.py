import numpy as np
true_w = np.array([1, 2, 3, 4, 5]) # Array of weights
d = len(true_w) #Length of weights
points = [] #Empty array
for i in range(100000): 
    x = np.random.randn(d) # d numbers are stored in x with gaussian distribution
    y = true_w.dot(x) + np.random.randn()
    points.append((x, y))
def F(w): #for Trainloss
    return sum((w.dot(x) - y)**2 for x, y in points) / len(points)
def dF(w): #For Trainloss's derivative
    return sum(2*(w.dot(x) - y) * x for x, y in points) / len(points)
def gradientDescent(F, dF, d): # ALgorithm to update w
# Gradient descent
    w = np.zeros(d) #Take w as zeros initially
    eta = 0.01 #Step-size
    for t in range(100):
        value = F(w)
        gradient = dF(w)
        w = w - eta * gradient # actually updating w
        print('iteration {}: w = {}, F(w) = {}'.format(t, w, value))
gradientDescent(F, dF, d)
