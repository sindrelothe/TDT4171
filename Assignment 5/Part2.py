import pickle
from tensorflow import keras
from typing import List

import numpy as np
import os


def useKeras():
    print(os.listdir())
    with open(file="Assignment 5/data/keras-data.pickle", mode="rb") as file:
        data = pickle.load(file)

    x_train: List[List[int]] = data["x_train"]
    y_train: List[int] = data["y_train"]
    x_test: List[List[int]] = data["x_test"]
    y_test: List[int] = data["y_test"]
    max_length: int = data["max_length"]
    vocab_size: int = data["vocab_size"]

    X_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)
    X_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 4))
    # model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=8))
    # model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=4, activation="softmax"))
    model.compile(
        # metrics=["accuracy"],
        # optimizer=keras.optimizers.Adam(learning_rate=1e-1, decay=1e-2),
        loss="sparse_categorical_crossentropy",
    )

    model.fit(X_train, np.array(y_train), epochs=1)
    model.save("model.h5")


useKeras()
