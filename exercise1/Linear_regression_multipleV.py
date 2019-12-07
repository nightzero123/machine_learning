#多特征的线性回归
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Size of the points dataset.
array = np.loadtxt('D:/data1.txt', delimiter=',')
m = array.size//3
cost = []
# Points x-coordinate and dummy value (x0, x1).
X0 = np.ones((m, 1))
X1 = np.array(array[:, 0]).reshape(m, 1)
X1 = (X1-np.mean(X1))/np.std(X1)  # 特征归一化
X2 = np.array(array[:, 1]).reshape(m, 1)
X2 = (X2-np.mean(X2))/np.std(X2)  # 特征归一化
X = np.hstack((X0, X1, X2))
# Points y-coordinate
y = np.array(array[:, 2]).reshape(m, 1)
y = (y-np.mean(y))/np.std(y)   # 特征归一化
# The Learning Rate alpha.
alpha = 0.01
epoch = 0
#
def error_function(theta, X, y):
	'''Error function J definition.'''
	diff = np.dot(X, theta) - y
	return (1./2*m) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
	'''Gradient of the function J definition.'''
	diff = np.dot(X, theta) - y #得到每个点的差别矩阵
	temp = np.sum(np.power(diff, 2))/m
	return (1./m) * np.dot(np.transpose(X), diff), temp

def gradient_descent(X, y, alpha):
	'''Perform gradient descent.'''
	theta = np.array([1, 1, 1]).reshape(3, 1)
	gradient, costt= gradient_function(theta, X, y)
	cost.append(costt)
	while not np.all(np.absolute(gradient) <= 1e-5):
		theta = theta - alpha * gradient
		gradient, costt = gradient_function(theta, X, y)
		cost.append(costt)
	return theta

def normalEqn(X,y):
	theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
	return theta.T

optimal = gradient_descent(X, y, alpha)
optimal1 = normalEqn(X, y)
print('optimal:', optimal)
print('optimal1:', optimal)
print('error function:', error_function(optimal, X, y)[0,0])

# 画图查看训练过程
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(len(cost)), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
'''
x = array[:, 0]
y = array[:, 1]
plt.scatter(x, y, alpha=0.6)
x1 = np.arange(0, 20, 0.1)
y1 = optimal[0]+optimal[1]*x1;
plt.plot(x1, y1)
plt.show()
'''
