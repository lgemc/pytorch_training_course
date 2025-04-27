import unittest

import torch

from data.datasets.tokenizer import  TokenizerDataset
from data.lib.split import train_test_split_sequential
from models.from_scratch import Basic
from from_scratch import train

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 140
model_dim = 64
heads_num = 8
blocks_num = 8

class TestFromScratch(unittest.TestCase):
    def test_init(self):
        train_set = TokenizerDataset(
            "../data/static/train.txt",
            batch_size_words=140,
            max_token_length=140,
            amount_of_samples=3000,
        )

        validation_set = TokenizerDataset(
            "../data/static/test.txt",
            batch_size_words=140,
            max_token_length=140,
            amount_of_samples=600,
        )

        model = Basic(
            vocab_size=train_set._vocab_size,
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
            epochs=30,
            batch_size=32,
            learning_rate=0.001,
            device=device,
            logging_steps=4,
            vocab_size=train_set.tokenizer.vocab_size
        )

        # store the model
        torch.save(model.state_dict(), "model.pth")
