import unittest

from torch.utils.data import DataLoader, RandomSampler
from tokenized import TokenizedDataset

class TestTokenizerDataset(unittest.TestCase):
    def test_init(self):
        # Test if the dataset initializes correctly
        dataset = TokenizedDataset("../static/test.txt", batch_size_chars=20)
        self.assertIsInstance(dataset, TokenizedDataset)
        self.assertEqual(dataset._vocab_size, 50257)
        print(dataset.tokenized)
        print(dataset._raw_content[:100])
        print(dataset[0])

    def test_random_sampler(self):
        dataset = TokenizedDataset(
            "../static/train.txt",
            batch_size_words=140,
            max_token_length=140,
        )
        print(dataset._indexes)
        dataloader = DataLoader(dataset, batch_size=2, sampler=RandomSampler(dataset))
        print(f"Dataset size: {len(dataset)}")

        epochs = 2
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            for batch in dataloader:
                x, _ = batch
                for x_i in x:
                    print(dataset.tokenizer.decode(x_i))
                    print("__________")