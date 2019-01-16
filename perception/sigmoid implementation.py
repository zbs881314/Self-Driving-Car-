import numpy as np
import matplotlib.pyplot as plt


def draw(x1, x2):
    ln = plt.plot(x1, x2)


def sigmoid(score):
    return 1/(1 + np.exp(-score))

n_pts = 100
np.random.seed(0)
bias = np.ones(n_pts)

top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
all_points = np.vstack((top_region, bottom_region))
#print(all_points)
w1 = -0.2
w2 = -0.35
b = 3.5
line_parameters = np.matrix([w1, w2, b]).T
#x1 = bottom_region[:, 0].min()
x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
x2 = -b / w2 + x1 * (-w1 / w2)
linear_combination = all_points*line_parameters
probabilities = sigmoid(linear_combination)
#print(probabilities)
#print(linear_combination)
#print(x2)
#print(all_points.shape)
#print(line_parameters.shape)
_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
draw(x1, x2)
plt.show()