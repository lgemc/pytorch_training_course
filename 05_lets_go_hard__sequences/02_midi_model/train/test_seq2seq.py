import unittest

import torch

from data.main import MIDIDataset
from train.seq2seq import train as train_seq2seq
from models.seq2seq import MusicSeq2SeqModel
from shared.datasets import train_test_split_sequential

device = "cuda" if torch.cuda.is_available() else "cpu"

class TestTrainSeq2Seq(unittest.TestCase):
    def test_train(self):
        data = MIDIDataset("../../stubs/chopin", sequence_size=5)

        self.assertIsNotNone(data)

        model = MusicSeq2SeqModel(
            hidden_dim=128,
            num_layers=8,
            pitch_vocab_size=128,
            velocity_vocab_size=128,
            pitch_embed_dim=32,
            velocity_embed_dim=32,
            dropout=0.3,
            step_max=data.normalized.max_step,
            step_min=data.normalized.min_step,
            duration_max=data.normalized.max_duration,
            duration_min=data.normalized.min_duration,
        )

        train_dataset, test_dataset = train_test_split_sequential(data, test_size=0.2)

        self.assertIsNotNone(model)
        train_seq2seq(
            model,
            train_dataset,
            test_dataset,
            epochs=200,
            device=device,
            batch_size=200,
            log_interval=1000,
            learning_rate=0.001,
        )

        torch.save(model.state_dict(), "seq2seq_model.pth")
