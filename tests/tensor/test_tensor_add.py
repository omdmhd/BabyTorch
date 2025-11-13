import unittest

from nn.tensor import Tensor
import numpy as np

class TestTensorSum(unittest.TestCase):
    def test_simple_sum(self):
        a = Tensor([1.0, 2.3], requires_grad=True)
        b = Tensor([2.0, 4.5], requires_grad=True)

        out = a + b

        np.array_equal(out.numpy(), [3.0, 6.8])

        out.backward()

        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)


    def test_sum_sepcial_case(self):
        a = Tensor([1.0, 2.3], requires_grad=True)

        out = a + a + a + a

        np.array_equal(out.numpy(), [4.0, 9.2])
        out.backward()
        np.array_equal(a.grad, np.array([4.0, 4.0]))

