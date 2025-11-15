import numpy as np
from nn.optimizer.base import BaseOptimizer
from nn.tensor import Tensor

class AdamOptimizer(BaseOptimizer):
    def __init__(self, parameters: list[Tensor], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.lr = Tensor([lr])
        self.beta1, self.beta2 = betas
        self.beta1 = Tensor([self.beta1])
        self.beta2 = Tensor([self.beta2])
        self.eps = Tensor([eps])
        self.weight_decay = weight_decay

        # Keep state in a list aligned with self.params to avoid hashing issues
        self.state = [{'step': 0, 'm': None, 'v': None} for _ in self.parameters]

    def step(self) -> None:
        for p, state in zip(self.parameters, self.state):
            if p.grad is None:
                continue
            g = p.grad
            if state['m'] is None:
                state['m'] = Tensor(np.zeros_like(p.data, dtype=np.float64))
                state['v'] = Tensor(np.zeros_like(p.data, dtype=np.float64))

            m = state['m']
            v = state['v']

            state['step'] += 1
            t = state['step']

            m = self.beta1 * m + (Tensor([1.0]) - self.beta1) * g
            v = self.beta2 * v + (Tensor([1.0]) - self.beta2) * (g * g)

            m_hat = m / (Tensor([1.0]) - self.beta1 ** t)
            v_hat = v / (Tensor([1.0]) - self.beta2 ** t)

            update = self.lr * m_hat / (v_hat.sqrt() + self.eps)
            p = p - update  # in-place replacement (or use -=)


    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = 0