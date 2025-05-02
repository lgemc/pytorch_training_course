import unittest

from sympy import sequence

from normalized import Normalized
from base import Base
from sequences import Sequenced

class TestSequences(unittest.TestCase):
    def setUp(self):
        # Load the dataset
        self.base = Base("../../stubs/chopin")
        self.normalized = Normalized(self.base)
        self.sequenced = Sequenced(self.normalized, sequence_size=5)

    def test_build_sequences(self):
        # Check if the sequences are built correctly
        self.assertIsNotNone(self.sequenced.sequences)
        self.assertEqual(2, len(self.sequenced.sequences[0]))
        self.assertEqual(5, len(self.sequenced.sequences[0][0]))

        print(self.sequenced.sequences[0][0])
        print(self.sequenced.sequences[0][1])
        print(self.sequenced.sequences[1][0])
        print(self.sequenced.sequences[1][1])


    def test_sequence_length(self):
        # Check if the sequence length is correct
        for seq in self.sequenced.sequences:
            self.assertEqual(len(seq), 5)