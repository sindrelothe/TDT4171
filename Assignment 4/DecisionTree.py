import numpy as np
from typing import List
import os


class DecisionTree:
    def __init__(self, example_filename: str, test_filename: str, path: str):
        self.examples: np.ndarray = np.genfromtxt(
            path + example_filename, delimiter=","
        )
        self.test_values: np.ndarray = np.genfromtxt(
            path + test_filename, delimiter=","
        )

    def randomImportance(self):
        pass

    def learn(
        self, examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray
    ):
        if self.examples.size == 0:
            return np.bincount(parent_examples).argmax()


def main():
    path = "Assignment 4/data/"
    test = "test.csv"
    train = "train.csv"
    tree: DecisionTree = DecisionTree(train, test, path)

    print(tree.examples.var())


if __name__ == "__main__":
    main()
