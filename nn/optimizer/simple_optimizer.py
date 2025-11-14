import numpy as np
from nn.optimizer.base import BaseOptimizer
from nn.tensor import Tensor

class SimpleOptimizer(BaseOptimizer):
    def __init__(self, parameters: list[Tensor], learning_rate: float):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self) -> None:
        for parameter in self.parameters:
            print("data", parameter.data.shape)
            print("grad", parameter.grad.shape)
            print("data", parameter.data)
            print("grad", parameter.grad)
            if len(parameter.grad.shape) > 0:
                parameter.data -= self.learning_rate * np.mean(parameter.grad, axis=0)
            else:
                parameter.data -= self.learning_rate * parameter.grad
            print("--------------------------------")    

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = 0