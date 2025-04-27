import unittest
import torch

from model import Model
from train import train, evaluate
from dataset import MNIST
from torchmetrics.classification import MulticlassF1Score
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
        # :self.assertTrue(os.path.exists("model.pth"))

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


    def test_f1_score(self):
        model = Model(hidden_size=36)
        model.load_state_dict(torch.load("model.pth"))
        model.eval()
        model.to(device)

        dataset = self.test_data

        x = torch.stack([dataset[i][0] for i in range(len(dataset))])
        y = torch.stack([dataset[i][1] for i in range(len(dataset))])
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            prediction = model(x)

        self.assertEqual(10000, len(prediction))

        _, prediction = torch.max(prediction, 1)

        scorer = MulticlassF1Score(num_classes=10).to(device)

        score = scorer(prediction, y)
        print(f"F1 score: {score}")
        self.assertGreater(score, 0.92)

