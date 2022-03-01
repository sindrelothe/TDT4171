import graphviz
import numpy as np
from typing import List, Dict, Callable, Set


class Node:
    def __init__(self):
        self.parent: Node = None
        self.children: Dict[int, Node] = {}
        self.attribute: int = None
        self.type: int = None

    def draw(self, filename):
        tree = graphviz.Digraph(name="Decision Tree", filename=filename)
        if self.type:
            tree.node(name=f"type_{self.type}", label=f"Type {self.type}")
            tree.render(view=True)
            return

        tree.node(name=f"attr_{self.attribute}", label=f"Attr {self.attribute}?")

        self.draw_child(tree, f"attr_{self.attribute}")

        tree.render(view=False)

    def draw_child(self, tree: graphviz.Digraph, parent_string: str):
        for key in self.children.keys():
            if self.children[key].type:
                child_name = f"type_{self.children[key].type}"
                child_label = f"Type {self.children[key].type}"
            else:
                child_name = f"attr_{self.children[key].attribute}{key}"
                child_label = f"Attr {self.children[key].attribute}?"
            child_name = parent_string + child_name
            tree.node(
                name=child_name,
                label=child_label,
            )
            tree.edge(
                tail_name=parent_string,
                head_name=child_name,
                label=f"{key}",
            )
            self.children[key].draw_child(tree, child_name)


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

    def classify(self, values: List[int]):
        if self.root.type:
            return self.root.type
        if len(self.root.children.keys()) == 0:
            raise Exception("Model is not trained")

        current: Node = self.root
        while not current.type:
            current = current.children[values[current.attribute]]

        return current.type

    def randomImportance(self, a: int, values: List[Data]):
        return np.random.random()

    def BFunction(self, q: float) -> float:
        return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))

    def remainder(self, a: int, values: List[Data]):
        values_set: Set[int] = set([d.attributes[a] for d in values])
        out = 0
        for att in values_set:
            pk = len([d.type for d in values if d.type == 1 and d.attributes[a] == att])
            nk = pk = len(
                [d.type for d in values if d.type == 2 and d.attributes[a] == att]
            )
            if pk > 0 or nk > 0:
                out += (pk + nk) * self.BFunction(pk / (pk + nk))
        out /= len(values)
        return out

    def maxInformationImportance(self, a: int, values: List[Data]):
        p = len([d.type for d in values if d.type == 1])
        return self.BFunction(p / (len(values))) - self.remainder(a, values)

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
            exs = [e for e in examples if e.attributes[a_next] == v]
            attributes_next = list(attributes.copy())
            attributes_next.remove(a_next)
            subtree: Node = self.learn(
                exs, np.array(attributes_next), examples, importanceFunc
            )
            root.children[v] = subtree
            subtree.parent = root

        return root

    def score(self, test_values: List[Data]):
        fasit = np.array([tr.type for tr in test_values])
        train_values = np.array(
            [self.classify(list(tr.attributes)) for tr in test_values]
        )
        difference: np.ndarray = np.abs(fasit - train_values)
        score = 1 - np.sum(difference) / difference.size
        return score


def main():
    path = "Assignment 4/data/"
    test = "test.csv"
    train = "train.csv"
    tree: DecisionTree = DecisionTree(train, test, path)
    root: Node = tree.learn(
        tree.data_examples_list,
        tree.attributes,
        [],
        tree.randomImportance,
    )
    root.draw("random_dt")
    tree.root = root

    accuracy = tree.score(tree.data_test_list)
    print(accuracy)
    root2: Node = tree.learn(
        tree.data_examples_list,
        tree.attributes,
        [],
        tree.maxInformationImportance,
    )
    root2.draw("maxImportance_dt")
    tree.root = root2

    accuracy = tree.score(tree.data_test_list)
    print(accuracy)


if __name__ == "__main__":
    main()
