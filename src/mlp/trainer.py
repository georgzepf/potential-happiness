import json
from pathlib import Path
from typing import Self

from .engine import MLP


class Trainer:
    PARAMETERS_DIR = "persisted_parameters"

    def __init__(self, model: MLP, x_train, y_train) -> None:
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def load_parameters(self, uid: str) -> Self:
        file = Path(f"{self.PARAMETERS_DIR}/{uid}.json")
        if file.exists():
            with open(file) as f:
                raw_parameters = json.load(f)

            model_parameters = self.model.parameters()

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

    def optimize_with_sse_loss(self, max_steps: int, target_loss: float, lr: float) -> Self:
        for step in range(max_steps):
            # forward pass
            y_pred = [self.model(train) for train in self.x_train]
            loss = sum(
                (pred - train) ** 2 for train, pred in zip(self.y_train, y_pred)
            )  # sum of squared errors (SSE)

            # backward pass
            self.model.zero_grad()
            loss.backward()

            # update parameters
            for p in self.model.parameters():
                p.data += -lr * p.grad

            if step % 1 == 0:
                print(f"step {step} loss {loss.data:.4f}")

            if loss.data <= target_loss:
                print(f"target_loss {target_loss} reached at step {step}")

                return self
        else:
            print(f"did not reach target_loss {target_loss} within max_steps {max_steps}")

        return self

    def persist_parameters(self, uid: str) -> Self:
        parent_dir = Path(self.PARAMETERS_DIR)
        parent_dir.mkdir(parents=True, exist_ok=True)

        file = parent_dir / f"{uid}.json"
        model_parameters = [p.data for p in self.model.parameters()]

        with open(file, "w") as f:
            json.dump(model_parameters, f)

        print(f"persisted parameters to {file.absolute()}")

        return self
