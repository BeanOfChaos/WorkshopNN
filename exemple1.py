"""Toy Keras API example
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


if __name__ == "__main__":
    xs = np.random.choice(100, size=(10000, 2))
    y = xs.sum(axis=1)

    model = Sequential()
    model.add(Dense(10, input_shape=(2,)))
    model.add(Dense(1))
    model.add(Activation("linear"))
    opt = Adam(lr=0.1)

    model.compile(optimizer=opt,
                  loss='mse')

    model.fit(xs, y,
              epochs=10,
              steps_per_epoch=10000 // 100
              )
    test = np.random.choice(100, size=(10, 2))
    print(test)
    print(model.predict(test))
