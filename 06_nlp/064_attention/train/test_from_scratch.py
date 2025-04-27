import unittest

import torch

from data.datasets.tokenizer import  TokenizerDataset
from data.lib.split import train_test_split_sequential
from models.from_scratch import Basic
from from_scratch import train

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 140
model_dim = 64
heads_num = 4
blocks_num = 3

class TestFromScratch(unittest.TestCase):
    def test_init(self):
        dataset = TokenizerDataset(
            "../data/static/input.txt",
            batch_size_words=140,
            max_token_length=140,
        )
        print(len(dataset[0]))
        #self.assertEqual(79, len(dataset))

        train_set, validation_set = train_test_split_sequential(dataset, 0.2)

        #self.assertEqual(451, len(train_set))
        #self.assertEqual(112, len(validation_set))

        print(dataset._vocab_size,)
        model = Basic(
            vocab_size=dataset._vocab_size,
            block_size=block_size,
            model_dim=model_dim,
            heads_num=heads_num,
            blocks_num=blocks_num,
            device=device
        )

        x, y = train_set[0]
        print(f"X: {x}")
        print(f"Y: {y}")

        train(
            model,
            train_dataset=train_set,
            val_dataset=validation_set,
            epochs=10,
            batch_size=block_size,
            learning_rate=0.0001,
            device=device,
            logging_steps=4,
            vocab_size=dataset.tokenizer.vocab_size
        )

        # store the model
        torch.save(model.state_dict(), "model.pth")
