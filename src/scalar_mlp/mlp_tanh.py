# implements the simple MLP described in notebooks/scalar_autograd+MLP.ipynb
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

from scalar_autograd import Value


class Module(ABC):
    @abstractmethod
    def parameters(self) -> Sequence[Value]:
        pass

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    def __init__(self, n_in: int) -> None:
        self.w = [
            # dividing each initial weight by math.sqrt(n_in) keeps pre-activation variance roughly constant,
            # regardless of layer width -> prevents tanh saturation
            Value(random.uniform(-1, 1) / math.sqrt(n_in))
            for _ in range(n_in)
        ]
        self.b = Value(0)

    # Neuron activation
    def __call__(self, x: Sequence[float | Value]) -> Value:
        act = sum(
            (wi * xi for wi, xi in zip(self.w, x)),
            self.b,  # passing self.b as the `start` param for sum(); same as `+ b`
        )
        return act.tanh()

    def parameters(self) -> Sequence[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"Neuron({len(self.w)})"


# in this simple MLP, a Layer processes multiple Neurons in parallel; no connection between them across the layer itself
class Layer(Module):
    # n_out = size of the layer's output vector, not "neuron count".
    # this way the constructor shape matches across all three modules (see comment in MLP class below)
    def __init__(self, n_in: int, n_out: int) -> None:
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x: Sequence[float | Value]) -> Sequence[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> Sequence[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLPtanh(Module):
    def __init__(self, n_in: int, n_outs: Sequence[int]) -> None:
        # explicit conversion, as list cannot be concatenated with type Sequence
        size = [n_in] + list(n_outs)

        # Layer(size[i], size[i + 1]) = pairwise sliding window over adjacent sizes
        # each layer's n_out IS the next layer's n_in
        # shapes match by construction, no separate check needed
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x: Sequence[float]) -> Sequence[Value]:
        activations: Sequence[float | Value] = x
        for layer in self.layers:
            activations = layer(activations)

        values: Sequence[Value] = cast(Sequence[Value], activations)
        return values

    def parameters(self) -> Sequence[Value]:
        return [p for l in self.layers for p in l.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"
