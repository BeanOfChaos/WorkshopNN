import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropoutkeras
from tensorflow.keras.optimizers import Adam


BATCH_SIZE = 16
VAL_SPLIT = 0.1

if __name__ == "__main__":

    # Datset documentation: https://www.tensorflow.org/datasets/catalog/titanic
    ds = tfds.load("titanic",
                   shuffle_files=True,
                   as_supervised=True,
                   )["train"]

    # lists are easier to manipulate, and no significative performance overhead
    # format: list of tuples as (dict of features, target)
    ds = list(ds.as_numpy_iterator())

    allf = {key: set() for key in ds[0][0].keys()}
    for elem in ds:
        for key in elem[0].keys():
            allf[key].add(elem[0][key])



    #  dataset preprocessing
    # Available features:
    #   - age (float32), boat (string), body (int32), cabin (string),
    #     embarked (int64), fare (float32), home.dest (string), name (string),
    #     parch (int32), pclass (int64), sex (int64), sibsp (int32),
    #     ticket (string)

    one_hot_encoding()
    convert_to_np_array()

    # build model
    model = Sequential()

    # fill in the blank

    model.add(Dense(1), activation='sigmoid')

    model.compile(
                  metrics=['accuracy'],
                  )


    # train model
    BATCH_SIZE = 16
    VAL_SPLIT = 0.1

    xs = np.asarray([elem[0] for elem in ds])
    ys = np.asarray([elem[1] for elem in ds])

    model.fit(x=xs, y=ys, batch_size=BATCH_SIZE,
              valdiation_split=VAL_SPLIT)
