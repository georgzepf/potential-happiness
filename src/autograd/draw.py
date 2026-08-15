from graphviz import Digraph


class Draw:
    def __init__(self, root):
        self._root = root

    def _trace(self):
        nodes, edges = set(), set()

        def build(parent):
            if parent not in nodes:
                nodes.add(parent)

                for child in parent._children:
                    edges.add((child, parent))

                    build(child)

        build(self._root)
        return nodes, edges

    def __call__(self):
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
        nodes, edges = self._trace()

        for node in nodes:
            node_uid = str(id(node))
            dot.node(
                name=node_uid,
                label=f"data {node.data:.4f} | grad {node.grad:.4f}",
                shape="record",
            )

            if node._op:
                dot.node(
                    name=node_uid + node._op, label=node._op
                )  # node showing the operation
                dot.edge(node_uid + node._op, node_uid)

        for node1, node2 in edges:
            dot.edge(str(id(node1)), str(id(node2)) + node2._op)

        return dot
