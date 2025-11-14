import numpy as np
import torch
from typing import Union

class Tensor:
    def __init__(self, data: Union[np.ndarray, torch.Tensor, list], requires_grad: bool = False):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, torch.Tensor):
            self.data = data.clone().detach()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        self._backward = lambda: None
        self.requires_grad = requires_grad
        self.children = []
        self.grad = 0


    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, children={len(self.children)}, grad={self.grad})"

    def numpy(self) -> np.ndarray:
        return self.data

    def __add__(self, other):
        out = Tensor(self.data + other.data)
        out.children.append(self)
        out.children.append(other)
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Tensor(self.data - other.data)
        out.children.append(self)
        out.children.append(other)
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other ** -1

    def __pow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            other = Tensor(np.array([other]))
        out = Tensor(self.data ** other.data)
        out.children.append(self)
        out.children.append(other)
        def _backward():
            self.grad += other.data * self.data ** (other.data -1) * out.grad
            other.grad += self.data ** other.data * np.log(self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data))
        out.children.append(self)
        def _backward():
            self.grad += out.grad * np.exp(self.data)
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)))
        out.children.append(self)
        def _backward():
            self.grad += out.grad * out.data * (1 - out.data)
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data))
        out.children.append(self)
        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Tensor(np.tanh(self.data))
        out.children.append(self)
        def _backward():
            self.grad += out.grad * (1 - out.data ** 2)
        out._backward = _backward
        return out
        
    def __mul__(self, other):
        out = Tensor(self.data * other.data)
        out.children.append(self)
        out.children.append(other)
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)
        out.children.append(self)
        out.children.append(other)
        def _backward():
            print(other.data.T.shape, out.grad.shape)
            self.grad += out.grad @other.data.T 
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.array([self.data.sum()]))
        out.children.append(self)
        def _backward():
            self.grad += out.grad * np.ones_like(self.data)
        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(np.array([self.data.mean()]))
        out.children.append(self)
        def _backward():
            self.grad += out.grad * np.ones_like(self.data) / self.data.size
        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0
        self.run_backward_recursively()

    def run_backward_recursively(self):
        self._backward()
        for child in self.children:
            child.run_backward_recursively()

    def T(self):
        return Tensor(self.data.T)