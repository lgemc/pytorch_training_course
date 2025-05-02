import unittest

from normalized import Normalized
from base import Base

class TestNormalized(unittest.TestCase):
    def setUp(self):
        # Load the dataset
        self.base = Base("../../stubs/chopin")
        self.normalized = Normalized(self.base)

    def test_normalize(self):
        # Check if the normalization was done correctly
        self.assertIsNotNone(self.normalized.data)
        self.assertEqual(85355, len(self.normalized.data))
        self.assertTrue("step" in self.normalized.data.columns)
        self.assertTrue("duration" in self.normalized.data.columns)

    def test_min_max_values(self):
        # Check if the min and max values are set correctly
        self.assertIsNotNone(self.normalized.min_step)
        self.assertIsNotNone(self.normalized.max_step)
        self.assertIsNotNone(self.normalized.min_duration)
        self.assertIsNotNone(self.normalized.max_duration)