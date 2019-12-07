#单特征的线性回归
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Size of the points dataset.
array = np.loadtxt('D:/data.txt', delimiter=',')
m = array.size//2

# Points x-coordinate and dummy value (x0, x1).
X0 = np.ones((m, 1), int)
X1 = np.array(array[:, 0]).reshape(m, 1)
X = np.hstack((X0, X1))

# Points y-coordinate
y = np.array(array[:, 1]).reshape(m, 1)

# The Learning Rate alpha.
alpha = 0.01

#
def error_function(theta, X, y):
	'''Error function J definition.'''
	diff = np.dot(X, theta) - y
	return (1./2*m) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
	'''Gradient of the function J definition.'''
	diff = np.dot(X, theta) - y #得到每个点的差别矩阵
	return (1./m) * np.dot(np.transpose(X), diff) 

def gradient_descent(X, y, alpha):
	'''Perform gradient descent.'''
	theta = np.array([1, 1]).reshape(2, 1)
	gradient = gradient_function(theta, X, y)
	while not np.all(np.absolute(gradient) <= 1e-5):
		theta = theta - alpha * gradient
		gradient = gradient_function(theta, X, y)
	return theta

optimal = gradient_descent(X, y, alpha)
print('optimal:', optimal)
print('error function:', error_function(optimal, X, y)[0,0])
x = array[:, 0]
y = array[:, 1]
plt.scatter(x, y, alpha=0.6)
x1 = np.arange(0, 20, 0.1)
y1 = optimal[0]+optimal[1]*x1;
plt.plot(x1, y1)
plt.show()
