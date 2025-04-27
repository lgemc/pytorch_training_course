import unittest

from torch.utils.data import DataLoader, RandomSampler
from tokenizer import TokenizerDataset

class TestTokenizerDataset(unittest.TestCase):
    def test_init(self):
        # Test if the dataset initializes correctly
        dataset = TokenizerDataset("../static/test.txt", batch_size_chars=20)
        self.assertIsInstance(dataset, TokenizerDataset)
        self.assertEqual(dataset._vocab_size, 50257)
        print(dataset.tokenized)
        print(dataset._raw_content[:100])
        print(dataset[0])

    def test_random_sampler(self):
        dataset = TokenizerDataset(
            "../static/test.txt",
            batch_size_words=5,
            max_token_length=5,
        )
        dataloader = DataLoader(dataset, batch_size=2, sampler=RandomSampler(dataset))
        print(f"Dataset size: {len(dataset)}")

        epochs = 2
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            for batch in dataloader:
                x, _ = batch
                for x_i in x:
                    print(dataset.tokenizer.decode(x_i))