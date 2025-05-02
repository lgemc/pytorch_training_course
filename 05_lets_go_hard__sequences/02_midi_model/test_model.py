import unittest
from models.model import MIDIModel
from data.dataset import MIDIDataset

midis_folder = "../stubs/chopin"

class TestModel(unittest.TestCase):
    def test_model(self):
        pitch_vocabulary_size = 128

        dataset = MIDIDataset(midis_folder)
        model = MIDIModel(pitch_vocab_size=pitch_vocabulary_size)

        model.eval()

        # Check if the model can process a batch of data
        x, y = dataset[0]
        output = model(x)

        self.assertIn("pitch", output)
        self.assertIn("velocity", output)
        self.assertIn("step", output)
        self.assertEqual(output["pitch"].shape[0], 128)