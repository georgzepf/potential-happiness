# implements the simple MLP described in `notebooks/autograd-MLP.ipynb`
import random

from autograd import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum(
            (wi * xi for wi, xi in zip(self.w, x)), self.b
        )  # passing self.b as the `start` param for sum(); same as `+ b`
        return act.tanh()

    def __repr__(self):
        return f"Neuron({len(self.w)})"
