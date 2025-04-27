import unittest

import torch
from transformers import GPT2Config

from data.datasets.tokenizer import TokenizerDataset
from data.lib.split import train_test_split_sequential
from models.fine_tune import FineTuneModel
from finetuned import train

device = "cuda" if torch.cuda.is_available() else "cpu"


class TestFineTuned(unittest.TestCase):
    def test_init(self):
        dataset = TokenizerDataset("../data/static/test.txt", batch_size_chars=600)
        self.assertEqual(66, len(dataset))

        train_set, validation_set = train_test_split_sequential(dataset, 0.2)

        #self.assertEqual(451, len(train_set))
        #self.assertEqual(112, len(validation_set))

        model = FineTuneModel(
            GPT2Config(),
            vocab_size=dataset._vocab_size,
        )

        train(
            model,
            train_dataset=train_set,
            val_dataset=validation_set,
            epochs=10,
            batch_size=8,
            learning_rate=0.0001,
            device=device,
            logging_steps=4,
        )
