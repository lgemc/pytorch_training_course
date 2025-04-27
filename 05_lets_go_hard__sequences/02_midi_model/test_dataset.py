import unittest

from dataset import MIDIDataset
from lib import NoteWithStep

midis_folder = "../stubs/chopin"

class TestDataset(unittest.TestCase):
    def test__init(self):
        dataset = MIDIDataset(midis_folder)
        self.assertEqual(48, len(dataset._midis))
        self.assertIsNotNone(dataset._ngrams)

        first_ngram = dataset._ngrams[0]

        self.assertEqual(2, len(first_ngram))
        self.assertEqual(5, len(first_ngram[0]))
        self.assertEqual(NoteWithStep, type(first_ngram[1]))
        self.assertEqual(NoteWithStep, type(first_ngram[0][0]))


    def test_get_item(self):
        dataset = MIDIDataset(midis_folder)
        x, y = dataset[0]

        self.assertEqual(5, len(x["step"]))
        self.assertEqual(5, len(x["velocity"]))
        self.assertEqual(5, len(x["pitch"]))

        first_ngram = dataset._ngrams[0]
        self.assertEqual(70, first_ngram[1].pitch)

        self.assertEqual(y["step"], first_ngram[1].step)
        self.assertEqual(y["velocity"], first_ngram[1].velocity)
        self.assertEqual(y["pitch"], first_ngram[1].pitch)

    def test_steps(self):
        dataset = MIDIDataset(midis_folder)

        self.assertEqual(85355, len(dataset.get_steps()))