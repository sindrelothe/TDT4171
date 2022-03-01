import graphviz

tree = graphviz.Digraph(name="Decision Tree", filename="example_dt")


tree.node(name="attr_1", label="Attr 1?")
tree.node(name="value_1", label="1")
tree.node(name="value_2", label="2")

tree.edge(tail_name="attr_1", head_name="value_1", label="2")
tree.edge(tail_name="attr_1", head_name="value_2", label="1")

tree.render(view=True)
