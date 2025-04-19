import unittest
from dataset import BasicOperationsDataset, decode_tokens

class TestDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = BasicOperationsDataset("dataset.txt")
        self.assertEqual(len(dataset), 152)
        x, y = dataset[0]
        self.assertEqual(decode_tokens(x), "1*7=")
        self.assertEqual(decode_tokens(y), "7")
        print(y)