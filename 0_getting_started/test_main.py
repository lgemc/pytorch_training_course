import unittest
import torch

class MainTest(unittest.TestCase):
    def test_main(self):
        self.assertTrue(torch.cuda.is_available())