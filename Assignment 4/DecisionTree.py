import numpy as np
from typing import List, Dict, Callable


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

    def randomImportance(self, a: int, values: List[Data]):
        return np.random.random()

    def BFunction(self, q: float) -> float:
        return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))

    def learn(
        self,
        examples: List[Data],
        attributes: List[int],
        parent_examples: List[Data],
        importanceFunc: Callable[[int, List[Data]], float],
    ) -> Node:
        if len(examples) == 0:
            node: Node = Node()
            node.type = np.bincount([p.type for p in parent_examples]).argmax()
            return node

        classifications: np.ndarray = np.array([ex.type for ex in examples])
        if classifications.var() == 0:
            node: Node = Node()
            node.type = classifications[0]
            return node

        if len(attributes) == 0:
            node: Node = Node()
            node.type = np.bincount([p.type for p in examples]).argmax()
            return node

        max_importance: float = -np.inf
        a_next = None
        for a in attributes:
            imp = importanceFunc(a, examples)
            if imp > max_importance:
                max_importance = imp
                a_next = a
        root = Node()
        root.attribute = a_next

        values_list: List[int] = []
        for ex in examples:
            if not ex.attributes[a_next] in values_list:
                values_list.append(ex.attributes[a_next])

        for v in values_list:
            exs = [e for e in examples if e.type == v]
            attributes_next = attributes.copy()
            attributes_next.remove(a_next)
            subtree: Node = self.learn(exs, attributes_next, examples)
            root.children[v] = subtree
            subtree.parent = root

        return root


def main():
    path = "Assignment 4/data/"
    test = "test.csv"
    train = "train.csv"
    tree: DecisionTree = DecisionTree(train, test, path)
    n: Data = Data(tree.examples[0])
    # for n in tree.data_examples_list:
    #     print(n)
    vals = np.array([1, 1, 1, 1])

    print(vals.var())


if __name__ == "__main__":
    main()
