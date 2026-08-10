import math


# implements the simple (scalar-based) autograd behavior described in notebooks/autograd+MLP.ipynb
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0  # dL/dself (this value's part of the gradient)

        self._children = set(
            _children
        )  # tuple for input convenience, set for performance
        self._op = _op

        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += out.grad  # accumulate, don't overwrite!
            other.grad += out.grad  # accumulate, don't overwrite!

        out._backward = backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * out.grad  # accumulate, don't overwrite!
            other.grad += self.data * out.grad  # accumulate, don't overwrite!

        out._backward = backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"

        out = Value(self.data ** other, (self,), f"**{other}")

        def backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad  # accumulate, don't overwrite!

        out._backward = backward

        return out

    def backward(self):
        self.grad = 1.0  # `backward` is called on assumably the loss function itself; therefore set grad to 1.0

        topo = []  # list keeps ordering and allows mutation
        visited = set()

        def build_topo(
                parent,
        ):  # build the ordered list `topo`; ensures each parent is only added after its children
            if parent not in visited:
                visited.add(parent)
                for child in parent._children:
                    build_topo(child)
                topo.append(parent)

        build_topo(self)

        for v in reversed(topo):
            v._backward()

    def tanh(
            self,
    ):  # takes a single real number and maps it to a range between -1 and 1
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        out = Value(t, (self,), "tanh")

        def backward():
            self.grad += (1 - t ** 2) * out.grad  # accumulate, don't overwrite!

        out._backward = backward

        return out

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
