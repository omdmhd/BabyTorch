import unittest

from nn.tensor import Tensor
import torch
import numpy as np

class TestTensorMul(unittest.TestCase):
    def test_simple_mul(self):

        torch_a = torch.tensor([2.], requires_grad=True)
        torch_b = torch.tensor([4.], requires_grad=True)
        torch_c = torch.tensor([6.], requires_grad=True)
        torch_d = torch.tensor([8.], requires_grad=True)

        torch_e = torch_a * torch_b
        torch_f = torch_c * torch_d
        torch_o = torch_e * torch_f

        a = Tensor([2.], requires_grad=True)
        b = Tensor([4.], requires_grad=True)

        c = Tensor([6.], requires_grad=True)
        d = Tensor([8.], requires_grad=True) 

        e = a * b
        d = c * d
        o = e * d

        np.array_equal(o.numpy(), torch_o.detach().numpy())

        o.backward()
        torch_o.backward()

        np.array_equal(a.grad, torch_a.grad.detach().numpy())
        np.array_equal(b.grad, torch_b.grad.detach().numpy())
        np.array_equal(c.grad, torch_c.grad.detach().numpy())
        np.array_equal(d.grad, torch_d.grad.detach().numpy())