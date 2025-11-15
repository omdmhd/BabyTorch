import numpy as np
import unittest


from nn.sequential import Sequential
from nn.layers.linear import LinearLayer
from nn.layers.sigmoid import Sigmoid
from nn.optimizer.simple_optimizer import SimpleOptimizer
from nn.optimizer.adam import AdamOptimizer
from nn.tensor import Tensor
from nn.module import Module
from nn.loss.mse import MSELoss

class TestBasicModel(unittest.TestCase):
    def test_basic_model(self):
        class BasicModel(Module):
            def __init__(self):
                super().__init__()
                self.layers = Sequential([
                    LinearLayer(2, 3),
                    Sigmoid(),
                    LinearLayer(3, 1),
                    Sigmoid(),
                ])
                self._parameters = self.layers._parameters

            def forward(self, x: np.ndarray) -> np.ndarray:
                logits = self.layers(x)
                return logits
                            
        X = Tensor(np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]))

        # Output data: XOR of inputs
        y = Tensor(np.array([
            [0],
            [1],
            [1],
            [0]
        ]))
        model = BasicModel()
        for epoch in range(0, 10):
            y_pred = model(X)
            loss_fn = MSELoss()
            loss = loss_fn(y_pred, y)
            optimizer = AdamOptimizer(model.parameters(), lr=0.01)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()