import unittest

import torch

from dataset import BasicOperationsDataset, vocab, train_test_split_sequential, decode_tokens
from model import MultiHeadAttention
from train import train

class TestTrain(unittest.TestCase):
    def test_train(self):
        dataset = BasicOperationsDataset("dataset.txt")
        model = MultiHeadAttention(vocab_size=len(vocab), num_heads=1)

        train_dataset, test_dataset = train_test_split_sequential(dataset, 0.2)

        train(model, train_dataset, test_dataset, epochs=2000)
        # torch.save(model.state_dict(), "model.pth")

    def test_predict(self):
        model = MultiHeadAttention(vocab_size=len(vocab), num_heads=1)

        model.load_state_dict(torch.load("model.pth"))

        dataset = BasicOperationsDataset("dataset.txt")
        train_dataset, test_dataset = train_test_split_sequential(dataset, 0.2)

        sample = test_dataset[5]
        x, y = sample
        print(f"y: {y}")
        print(decode_tokens(x))
        x = x.unsqueeze(0)
        print(x)

        model.eval()
        out = model(x)
        print(torch.nn.functional.log_softmax(out, dim=1))
        print(torch.argmax(out.squeeze(0)))