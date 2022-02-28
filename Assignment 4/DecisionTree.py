import numpy as np
from typing import List


class Node:
    def __init__(self, definition: np.ndarray):
        self.type: int = definition[-1]
        self.attributes: np.ndarray = definition[0:-1]
        self.parent: Node = None
        self.children: List[Node] = []

    def __repr__(self):
        return f"type={self.type}, attributes={self.attributes}"


class DecisionTree:
    def __init__(self, example_filename: str, test_filename: str, path: str):
        self.examples: np.ndarray = np.genfromtxt(
            path + example_filename, delimiter=",", dtype="int64"
        )
        self.test_values: np.ndarray = np.genfromtxt(
            path + test_filename, delimiter=",", dtype="int64"
        )

        self.node_examples_list: List[Node] = [Node(ex) for ex in self.examples]
        self.node_test_list: List[Node] = [Node(test) for test in self.test_values]

        attributes: List[int] = [
            n for n in range(len(self.node_examples_list[0].attributes))
        ]

    def randomImportance(self):
        pass

    def BFunction(self, q: float) -> float:
        return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))

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
    n: Node = Node(tree.examples[0])
    for n in tree.node_examples_list:
        print(n.attributes)


if __name__ == "__main__":
    main()
