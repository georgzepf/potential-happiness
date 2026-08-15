from typing import Callable

LrSchedule = Callable[[int, int], float]


def constant(lr: float) -> LrSchedule:
    return lambda step, max_steps: lr


def linear_decay(start: float, end: float) -> LrSchedule:
    def compute_lr(step: int, max_steps: int) -> float:
        return start - (start - end) * step / max_steps

    return compute_lr
