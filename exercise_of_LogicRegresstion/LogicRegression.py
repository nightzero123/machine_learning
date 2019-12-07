import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Size of the points dataset.
array = np.loadtxt('ex2data1.txt', delimiter=',')
m = array.size//3

# Points x-coordinate and dummy value (x0, x1).
X0 = np.ones((m, 1), int)
X1 = np.array(array[:, 0]).reshape(m, 1)
X2 = np.array(array[:, 1]).reshape(m, 1)
X = np.hstack((X0, X1, X2))

# Points y-coordinate
y = np.array(array[:, 2]).reshape(m, 1)

# The Learning Rate alpha.
alpha = 0.01


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def error_function(theta, X, y):
    '''Error function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./2*m) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
	'''Gradient of the function J definition.'''
	diff = sigmoid(np.dot(X, theta)) - y
	return (1./m) * np.dot(np.transpose(X), diff)

def gradient_descent(X, y, alpha):
	'''Perform gradient descent. '''
	theta = np.array([1, 1 ,1]).reshape(3, 1)
	gradient = gradient_function(theta, X, y)
	for i in range(1, 1000000):
		theta = theta - alpha * gradient
		gradient = gradient_function(theta, X, y)
	return theta


optimal = gradient_descent(X, y, alpha)
print('optimal:', optimal)
print('error function:', error_function(optimal, X, y)[0,0])
x = array[:, 0]
y = array[:, 1]
result = array[:,2]
mask = result > 0
mask1 = result <1
plt.scatter(x[mask], y[mask], c='b', alpha=0.6)
plt.scatter(x[mask1], y[mask1], c='g', alpha=0.6)

x1 = np.arange(20, 100, 0.1)
x2 = -1*optimal[0]/optimal[2]-optimal[1]/optimal[2]*x1
plt.plot(x1, x2)
plt.show()