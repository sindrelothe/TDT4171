from tensorflow import keras
import pickle
from typing import List
import numpy as np
import os

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

model = keras.models.load_model("model.h5")

res = model.evaluate(X_test, np.array(y_test))

print(res)
