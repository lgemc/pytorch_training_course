import unittest

import torch

from loss import cross_entropy

class TestLoss(unittest.TestCase):
    def test_array_indexing(self):
        """
        Array indexing test to show the advanced indexing in pytorch, which is used in cross entropy loss function
        to get the real logits indexed by the labels
        :return:
        """
        logits_example = torch.tensor([
            [-1, 1, 2],
            [0.7, 2, 3],
            [4, 6, 8]
        ])

        y_values = torch.tensor([0, 1, 2])
        self.assertTrue(torch.all(logits_example[range(len(y_values)), y_values] == torch.tensor([-1, 2, 8])))


    def test_cross_entropy__perfect_score(self):
        logits_example = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        y = torch.tensor([1, 0, 2])

        loss = cross_entropy(logits_example, y)

        std_loss = torch.nn.functional.cross_entropy(logits_example, y)
        self.assertTrue(loss.item(), std_loss.item())