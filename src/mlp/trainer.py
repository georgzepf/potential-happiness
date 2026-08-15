import json
from pathlib import Path
from typing import Self

from .loss import LossFunction
from .lr import LrSchedule, constant
from .mlp_tanh import MLP

REPO_ROOT = Path(__file__).resolve().parents[2]


class Trainer:
    def __init__(self, model: MLP, loss_fn: LossFunction, x_train, y_train, model_name: str) -> None:
        self._model = model
        self.loss_fn = loss_fn

        self._x_train = x_train
        self._y_train = y_train

        self._model_name = model_name

    def optimize(self, max_steps: int, target_loss: float, lr: LrSchedule = constant(0.01)) -> Self:
        for step in range(max_steps):
            # forward pass
            y_pred = [self._model(train)[0] for train in self._x_train]
            loss = self.loss_fn(y_pred, self._y_train)

            # backward pass
            self._model.zero_grad()
            loss.backward()

            # update parameters
            for p in self._model.parameters():
                current_lr = lr(step, max_steps)
                p.data += -current_lr * p.grad

            if step % 1 == 0:
                print(f"step {step} loss {loss.data:.4f}")

            if loss.data <= target_loss:
                print(f"target_loss {target_loss} reached at step {step}")

                return self
        else:
            print(f"did not reach target_loss {target_loss} within max_steps {max_steps}")

        return self

    def evaluate(self, x_test, y_test) -> Self:
        y_pred = [self._model(test)[0] for test in x_test]

        loss = self.loss_fn(y_pred, y_test)
        print(f"loss {loss.data:.4f}")

        correct = sum(
            1 for pred, test in zip(y_pred, y_test)
            if (pred.data > 0) == (test > 0)
        )
        accuracy = correct / len(y_test)
        print(f"accuracy {accuracy:.2%}")

        return self

    def load_parameters(self) -> Self:
        file = Path(f"{REPO_ROOT}/_parameters/{self._model_name}.json")
        if file.exists():
            with open(file) as f:
                raw_parameters = json.load(f)

            model_parameters = self._model.parameters()

            assert len(raw_parameters) == len(model_parameters), (
                f"model architecture mismatch!\n"
                f"raw parameters in file {len(raw_parameters)}, real parameters in model {len(model_parameters)}"
            )
            # attention: this approach matches purely based on the position of each parameter
            # loading parameters therefore only works if they were persistend from the same model size/architecture
            for mp, rp in zip(model_parameters, raw_parameters):
                mp.data = rp

            print(f"loaded parameters from {file.absolute()}")
        else:
            print(f"no file found at {file.absolute()}, skipping")

        return self

    def persist_parameters(self) -> Self:
        parent_dir = Path(REPO_ROOT) / "_parameters"
        parent_dir.mkdir(parents=True, exist_ok=True)

        file = parent_dir / f"{self._model_name}.json"
        model_parameters = [p.data for p in self._model.parameters()]

        with open(file, "w") as f:
            json.dump(model_parameters, f)

        print(f"persisted parameters to {file.absolute()}")

        return self
