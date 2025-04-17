import unittest
import os

import torch

from model import Model
from train import train
from dataset import MNIST

root_folder = "stubs"
device = "cuda" if torch.cuda.is_available() else "cpu"

class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = MNIST(root_folder, split="test")
        cls.train_data = MNIST(root_folder, split="train")

    def test_train_and_store(self):
        model = Model(hidden_size=36)
        train(model, self.train_data, self.test_data, device=device, batch_size=256)

        # torch.save(model.state_dict(), "model.pth")
        # self.assertTrue(os.path.exists("model.pth"))

    def test_eval(self):
        model = Model(hidden_size=36)
        model.load_state_dict(torch.load("model.pth"))
        model.eval()

        # tenth test sample:
        x, y = self.test_data[10]
        print(f"Class: {y}")

        output = model(x)
        output_value = torch.argmax(output)
        print(f"Predicted value: {output_value}")
        self.assertEqual(y.item(), output_value.item())

