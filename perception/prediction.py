import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts), np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts), np.random.normal(6, 2, n_pts)]).T

X = np.vstack((Xa, Xb))
y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
#plt.ion()
plt.show()


model = Sequential()
model.add(Dense(units = 1, input_shape = (2,), activation = 'sigmoid'))
adam = Adam(lr = 0.1)
model.compile(adam, loss='binary_crossentropy', metrics = ['accuracy'])
h = model.fit(x=X, y=y, verbose=1, batch_size=50, epochs=500, shuffle='true')


# plt.plot(h.history['acc'])
# plt.title('accuracy')
# plt.xlabel('epoch')
# plt.legend(['accuracy'])
# plt.show()


# plt.plot(h.history['loss'])
# plt.title('loss')
# plt.xlabel('epoch')
# plt.legend(['loss'])
# plt.show()

# print(X.shape)
def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1)
    y_span = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1)
    # print(x_span)
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

x = 7.5
y = 5
point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color='red')
plt.show()
print('Prediction is:', prediction)