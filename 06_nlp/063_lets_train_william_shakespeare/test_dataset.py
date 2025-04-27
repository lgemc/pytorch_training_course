import unittest

from dataset import WilliamDataset

class TestDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = WilliamDataset("train.txt")
        self.assertEqual(len(dataset._vocabulary), 15052)

        print([dataset._tokenized[i] for i in range(20)])
        print(dataset._encoded_content[:20])
        print(dataset.decode(dataset._encoded_content[:20]))