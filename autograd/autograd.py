class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data

        self._children = set(
            _children
        )  # tuple for input convenience, set for performance
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, _children={self._children}, _op={self._op})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out


from graphviz import Digraph


class Draw:
    def __init__(self, root):
        self.root = root

    def _trace(self):
        nodes, edges = set(), set()

        def build(parent):
            if parent not in nodes:
                nodes.add(parent)

                for child in parent._children:
                    edges.add((child, parent))

                    build(child)

        build(self.root)
        return nodes, edges

    def __call__(self):
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
        nodes, edges = self._trace()

        for node in nodes:
            node_uid = str(id(node))
            dot.node(name=node_uid, label=str(node.data), shape="record")

            if node._op:
                dot.node(
                    name=node_uid + node._op, label=node._op
                )  # node showing the operation
                dot.edge(node_uid + node._op, node_uid)

        for node1, node2 in edges:
            dot.edge(str(id(node1)), str(id(node2)) + node2._op)

        return dot
