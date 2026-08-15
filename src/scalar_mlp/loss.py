from typing import Callable, Sequence

from scalar_autograd import Value

LossFunction = Callable[[Sequence[Value], Sequence[float]], Value]


# Sum of Squared Errors (SSE)
def sse(y_pred: Sequence[Value], y_gt: Sequence[float]) -> Value:
    return sum(
        (pred - gt) ** 2 for pred, gt in zip(y_pred, y_gt)
    )


# Mean Squared Error (MSE)
def mse(y_pred: Sequence[Value], y_gt: Sequence[float]) -> Value:
    return sum(
        (pred - gt) ** 2 for pred, gt in zip(y_pred, y_gt)
    ) / len(y_pred)
