from tensorflow import keras
import pickle

with open(file="Assignment 5/data/scikit-learn-data.pickle", mode="rb") as file:
    data = pickle.load(file)

print(data)
