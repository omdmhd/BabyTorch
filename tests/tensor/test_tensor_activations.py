import unittest

import numpy as np

from nn.tensor import Tensor


class TestTensorActivations(unittest.TestCase):
    def test_exp(self):
        x = Tensor([1.0, 0.0], requires_grad=True)

        out = x.exp()

        np.testing.assert_allclose(out.numpy(), np.exp(x.numpy()))

        out.backward()

        np.testing.assert_allclose(x.grad, np.exp(x.numpy()))

    def test_sigmoid(self):
        x = Tensor([-1.0, 0.5], requires_grad=True)

        out = x.sigmoid()

        expected = 1 / (1 + np.exp(-x.numpy()))
        np.testing.assert_allclose(out.numpy(), expected)

        out.backward()

        np.testing.assert_allclose(x.grad, expected * (1 - expected))

    def test_log(self):
        x = Tensor([1.0, 3.0], requires_grad=True)

        out = x.log()

        expected = np.log(x.numpy())
        np.testing.assert_allclose(out.numpy(), expected)

        out.backward()

        np.testing.assert_allclose(x.grad, 1 / x.numpy())

    def test_tanh(self):
        x = Tensor([-1.0, 2.0], requires_grad=True)

        out = x.tanh()

        expected = np.tanh(x.numpy())
        np.testing.assert_allclose(out.numpy(), expected)

        out.backward()

        np.testing.assert_allclose(x.grad, 1 - expected**2)

