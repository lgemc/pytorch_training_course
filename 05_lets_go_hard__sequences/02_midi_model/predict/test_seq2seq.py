import unittest

import torch

from data.main import MIDIDataset
from models.seq2seq import MusicSeq2SeqModel
from predict.seq2seq import predict, items_to_notes
from shared.midi.files import export

chopin_folder = "../../stubs/chopin"

device = "cuda" if torch.cuda.is_available() else "cpu"

class TestPredict(unittest.TestCase):
    def test_predict(self):
        sequence_size = 5
        dataset = MIDIDataset(chopin_folder, sequence_size=sequence_size)
        seed = dataset[0][0]

        model = MusicSeq2SeqModel(
            hidden_dim=128,
            num_layers=8,
            pitch_vocab_size=128,
            velocity_vocab_size=128,
            pitch_embed_dim=32,
            velocity_embed_dim=32,
            step_input_dim=1,
            step_max=dataset.normalized.max_step,
            step_min=dataset.normalized.min_step,
            duration_max=dataset.normalized.max_duration,
            duration_min=dataset.normalized.min_duration,
        )

        model.eval()
        model.load_state_dict(torch.load("../train/seq2seq_model.pth"))

        notes = predict(
            model=model,
            notes_amount=300,
            device=device,
            seed=seed,
        )

        # Convert the notes to pretty_midi.Note objects
        notes = items_to_notes(
            notes,
            step_max=dataset.normalized.max_step,
            step_min=dataset.normalized.min_step,
            duration_max=dataset.normalized.max_duration,
            duration_min=dataset.normalized.min_duration,
        )

        export(notes, "test.mid")


