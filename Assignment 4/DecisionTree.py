import numpy as np
from typing import List, Dict


class Node:
    def __init__(self):
        self.parent: Node = None
        self.children: Dict[int, Node] = {}
        self.attribute: int = None
        self.type: int = None


class Data:
    def __init__(self, definition: np.ndarray):
        self.type: int = definition[-1]
        self.attributes: np.ndarray = definition[0:-1]

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

        self.data_examples_list: List[Data] = [Data(ex) for ex in self.examples]
        self.data_test_list: List[Data] = [Data(test) for test in self.test_values]

        self.attributes: List[int] = [
            n for n in range(len(self.data_examples_list[0].attributes))
        ]

        self.root: Node = Node()

    def randomImportance(self, values):
        return np.random.choice(values)

    def BFunction(self, q: float) -> float:
        return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))

    def learn(
        self, examples: List[Data], attributes: List[int], parent_examples: List[Data]
    ):
        if len(examples) == 0:
            return np.bincount([p.type for p in parent_examples]).argmax()


def main():
    path = "Assignment 4/data/"
    test = "test.csv"
    train = "train.csv"
    tree: DecisionTree = DecisionTree(train, test, path)
    n: Data = Data(tree.examples[0])
    for n in tree.data_examples_list:
        print(n)


if __name__ == "__main__":
    main()
