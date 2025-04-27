import unittest
from torch.utils.data import DataLoader

from basic import Dataset

class TestBasicDataset(unittest.TestCase):
    def test_init(self):
        # Test if the dataset initializes correctly
        dataset = Dataset("../static/test.txt")
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(dataset._vocab_size, 78)

    def test_encode_decode_sequence(self):
        # Test if the encoding works correctly
        dataset = Dataset("../static/test.txt")
        encoded = dataset._encode_sequence(["What", " is", " Lorem"])
        expected_encoded = [69, 29, 8]
        self.assertEqual(encoded, expected_encoded)

        # Test if the decoding works correctly
        decoded = dataset._decode_sequence(encoded)
        expected_decoded = ["What", " is", " Lorem"]
        self.assertEqual(decoded, expected_decoded)

    def test_random_sampler(self):
        dataset = Dataset("../static/test.txt", sequence_len=10)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        print(f"Dataset size: {len(dataset)}")

        epochs = 2
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            for batch in dataloader:
                print(batch)