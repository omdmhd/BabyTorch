from nn.module import Module
from nn.tensor import Tensor

class BaseLoss(Module):
    def __init__(self):
        super().__init__()

    def loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        raise NotImplementedError