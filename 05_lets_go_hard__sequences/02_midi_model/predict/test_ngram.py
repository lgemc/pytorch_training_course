import unittest

import torch

from data.dataset import MIDIDataset
from models.model import MIDIModel
from predict.predict import predict
from shared.midi.midi import write_midi

chopin_folder = "../stubs/chopin"

device = "cuda" if torch.cuda.is_available() else "cpu"

class TestPredict(unittest.TestCase):
    def test_predict(self):
        ngram_size = 5
        dataset = MIDIDataset(chopin_folder)
        seed = dataset._notes[:ngram_size]

        model = MIDIModel(
            pitch_vocab_size=128,
            velocity_vocab_size=128,
            step_input_dim=1,
            step_max=dataset.step_max,
            step_min=dataset.step_min,
            duration_max=dataset.duration_max,
            duration_min=dataset.duration_min,
        )

        model.eval()
        model.load_state_dict(torch.load("../model.pth"))

        notes = predict(
            model=model,
            notes_amount=30,
            device=device,
            seed=seed
        )

        self.assertEqual(len(notes), len(seed) + 30)

        write_midi(notes, "test.mid")


