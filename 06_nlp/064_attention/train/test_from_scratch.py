import unittest

import torch

from data.datasets.tokenized import  TokenizedDataset
from models.from_scratch import TransformerShakespeare
from from_scratch import train
from generate import generate_text

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 140
model_dim = 64
heads_num = 8
blocks_num = 8

class TestFromScratch(unittest.TestCase):
    def test_init(self):
        train_set = TokenizedDataset(
            "../data/static/train.txt",
            batch_size_words=140,
            max_token_length=140,
            amount_of_samples=3000,
        )

        validation_set = TokenizedDataset(
            "../data/static/test.txt",
            batch_size_words=140,
            max_token_length=140,
            amount_of_samples=600,
        )

        model = TransformerShakespeare(
            vocab_size=train_set._vocab_size,
            block_size=block_size,
            model_dim=model_dim,
            heads_num=heads_num,
            blocks_num=blocks_num,
            dropout=.3,
            device=device,
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
        torch.save(model.state_dict(), "../../../05_lets_go_hard__sequences/model.pth")