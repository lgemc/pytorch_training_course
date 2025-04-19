import unittest

import torch

from binary_operations import MULTIPLICATION, SUM, REST, get_operation_result, BinaryOperation

class TestBinaryOperations(unittest.TestCase):
    def test_sum(self):
        a = torch.tensor([1, 2, 3, 4, 5, 6])
        b = torch.tensor([2, 3, 4, 5, 6, 7])

        c = a + b

        ops = []
        for i, a_i in enumerate(a):
            ops.append(BinaryOperation(a_i.item(), b[i].item(), SUM))

        results = []

        for op in ops:
            results.append(get_operation_result(op))

        results = torch.tensor(results)
        self.assertTrue(torch.all(results == c))

    def test_multiplication(self):
        a = torch.tensor([1, 2, 3, 4, 5, 6])
        b = torch.tensor([2, 3, 4, 5, 6, 7])

        c = a * b

        ops = []
        for i, a_i in enumerate(a):
            ops.append(BinaryOperation(a_i.item(), b[i].item(), MULTIPLICATION))

        results = []

        for op in ops:
            results.append(get_operation_result(op))

        results = torch.tensor(results)
        self.assertTrue(torch.all(results == c))


    def test_rest(self):
        a = torch.tensor([1, 2, 3, 4, 5, 6])
        b = torch.tensor([2, 3, 4, 5, 6, 7])

        c = a - b

        ops = []
        for i, a_i in enumerate(a):
            ops.append(BinaryOperation(a_i.item(), b[i].item(), REST))

        results = []

        for op in ops:
            results.append(get_operation_result(op))

        results = torch.tensor(results)
        self.assertTrue(torch.all(results == c))