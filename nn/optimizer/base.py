from nn.tensor import Tensor

class BaseOptimizer:
    def __init__(self):
        pass

    def step(self, parameters: list[Tensor]) -> None:
        raise NotImplementedError