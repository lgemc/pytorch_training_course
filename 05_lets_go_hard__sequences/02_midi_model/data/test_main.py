import unittest
from data.main import MIDIDataset

class TestMain(unittest.TestCase):
    def test_midi_dataset(self):
        dataset = MIDIDataset("../../stubs/chopin", sequence_size=5)
        self.assertIsNotNone(dataset)
        self.assertEqual(84855, len(dataset))

        print(dataset[56])
