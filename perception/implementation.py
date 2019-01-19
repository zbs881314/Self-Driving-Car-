import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


np.random.seed(0)
n_pts = 500
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)

print(X)
print(y)


plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()


model = Sequential()
model.add(Dense(4, input_shape=(2, ), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(lr = 0.01), 'binary_crossentropy', metrics=['accuracy'])
h = model.fit(x=X, y=y, verbose=1, batch_size=20, epochs=100, shuffle='true')

# plt.plot(h.history['acc'])
# plt.legend(['accuracy'])
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
#
#
# plt.plot(h.history['loss'])
# plt.legend(['loss'])
# plt.ylabel('loss')
# plt.xlabel('epoch')


def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25)
    y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25)
    print(x_span)
    # print(y_span)
    xx, yy = np.meshgrid(x_span, y_span)
    # print(xx)
    # print(yy)
    xx_, yy_ = xx.ravel(), yy.ravel()
    # print(xx_)
    # print(yy_)
    grid = np.c_[xx_, yy_]
    # print(grid.shape)
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)
    #plt.ion()
    #plt.show()







plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
#plt.show()

x = 0
y = 0.75
point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color='red')
plt.show()
print('Prediction is:', prediction)

