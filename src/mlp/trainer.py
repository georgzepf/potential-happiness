from .engine import MLP


class Trainer:
    def __init__(self, model: MLP, x_train, y_train) -> None:
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def optimize_with_sse_loss(self, max_steps: int, target_loss: float, start_lr: float, end_lr: float) -> bool:
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
                lr = start_lr - ((start_lr - end_lr) * step / max_steps)
                p.data += -lr * p.grad

            if step % 1 == 0:
                print(f"step {step} loss {loss.data:.4f}")

            if loss.data <= target_loss:
                print(f"target_loss {target_loss} reached at step {step}")
                return True
        else:
            print(f"did not reach target_loss {target_loss} within max_steps {max_steps}")
            return False
