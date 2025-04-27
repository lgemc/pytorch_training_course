import unittest
import torch
from dataset import MNIST
from model import Model

from libs.images import print_image_as_char

root_folder = "stubs"

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load once for all test methods
        train = MNIST(root_folder, split="train")
        cls.train = train
        test = MNIST(root_folder, split="test")
        cls.test = test

    def test_train_split(self):
        self.assertEqual(60000, len(self.train.labels))
        self.assertEqual((60000, 28*28), self.train.data.shape)
        self.assertEqual(10000, len(self.test))

    def test_init_module(self):
        model = Model(hidden_size=32)
        print(model)

    def test_forward_pass(self):
        model = Model(hidden_size=32)
        x, y = self.test[0]
        result = model(x)
        print_image_as_char(x, 28, 28)
        self.assertEqual(10, len(result))
        result_number = torch.argmax(result)
        print(f"Real y: {y}, predicted y: {result_number}") # the model is not trained, so the result may be incorrect or not
