import numpy as np
array = np.loadtxt('ex2data2.txt', delimiter=',')
print(array)
X = np.array(array[:, 0:-1])  # 2创建矩阵X和y
y = np.array(array[:, -1])
print(X)
print(y)

