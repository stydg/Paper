
import math

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

# Normal distributed x and y vector with mean 0 and standard deviation 1
x = np.random.normal(0, 1, 500) #평균 0 표준편차 1
y = np.random.normal(0, 1, 500)
X = np.vstack((x, y)).T

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], s=2.5)
plt.title('Generated Data')
plt.axis('equal')


# Covariance
def cov(x, y):
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)

# Covariance matrix
def cov_mat(X):
    return np.array([[cov(X[0], X[0]), cov(X[0], X[1])], \
                     [cov(X[1], X[0]), cov(X[1], X[1])]])

# Calculate covariance matrix
cov_mat(X.T) # (or with np.cov(X.T))
print(cov_mat(X))



# 행렬을 원점에 중심
X = X - np.mean(X, 0)

# Scaling matrix
sx, sy = 3.4, 0.7
Scale = np.array([[sx, 0], [0, sy]])

# X에 스케일링 행렬 적용
Y = X.dot(Scale)

plt.subplot(1, 3, 2)
plt.scatter(Y[:, 0], Y[:, 1], s=2.5)
plt.title('Scaled Data')
plt.axis('equal')


# Scaling matrix
sx, sy = 3.4, 0.7
Scale = np.array([[sx, 0], [0, sy]])


#  회전 행렬
theta = 0.77*np.pi
theta = math.radians(-39)
c, s = np.cos(theta), np.sin(theta)
Rot = np.array([[c, -s], [s, c]])

# 변환 행렬
T = Scale.dot(Rot)

# Apply transformation matrix to X
Y = X.dot(T)

plt.subplot(1, 3, 3)
plt.scatter(Y[:, 0], Y[:, 1], s=2.5)
plt.title('Transformed Data')
plt.axis('equal');
plt.show()
# Calculate covariance matrix
cov_mat(Y.T)

