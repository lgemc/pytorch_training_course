import unittest

import torch

from dataset import BasicOperationsDataset, vocab, train_test_split_sequential, decode_tokens
from model import MultiHeadAttention
from train import train

device = "cuda" if torch.cuda.is_available() else "cpu"

class TestTrain(unittest.TestCase):
    def test_train(self):
        dataset = BasicOperationsDataset("dataset.txt")
        model = MultiHeadAttention(vocab_size=len(vocab), num_heads=1, dropout=0.2)

        train_dataset, test_dataset = train_test_split_sequential(dataset, 0.2)

        train(model, train_dataset, test_dataset, epochs=2000, batch_size=16, lr=0.001, device=device)
        torch.save(model.state_dict(), "model.pth")

    def test_predict(self):
        model = MultiHeadAttention(vocab_size=len(vocab), num_heads=1)

        model.load_state_dict(torch.load("model.pth"))

        dataset = BasicOperationsDataset("dataset.txt")
        train_dataset, test_dataset = train_test_split_sequential(dataset, 0.2)

        sample = [test_dataset[i] for i in range(10)]
        x = torch.stack([s[0] for s in sample])
        print([decode_tokens(xi) for xi in x])
        model.eval()
        out = model(x)
        print(torch.argmax(out, dim=-1))