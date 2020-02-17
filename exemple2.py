"""Regression over-fitting example
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

DS_SIZE = 1000
BATCH_SIZE = 250
N_EPOCHS = 1000
LR = 0.1
DENSE_UNITS = 30


if __name__ == "__main__":
    xs = np.random.random_sample(size=(DS_SIZE, 1)) * 2 * np.pi
    ys = np.sin(xs)

    model = Sequential()
    model.add(Dense(DENSE_UNITS, activation="tanh", input_shape=(1,)))
    model.add(Dropout(0))
    model.add(Dense(1))
    model.add(Dropout(0))
    model.add(Activation("linear"))

    opt = Adam(lr=LR)

    model.compile(optimizer=opt,
                  loss='mse',
                  metrics=["mae"])

    model.fit(xs, ys,
              epochs=N_EPOCHS,
              steps_per_epoch=DS_SIZE//BATCH_SIZE,
              batch_size=BATCH_SIZE,
              )

    test_x = np.arange(0, 15, 0.1)
    test_y = np.sin(test_x)

    pred_y = model.predict(test_x)

    plt.plot(test_x, test_y, label="'Real' values")
    plt.plot(test_x, pred_y, label="Predicted values")
    plt.legend()
    plt.show()
