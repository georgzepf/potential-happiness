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
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
