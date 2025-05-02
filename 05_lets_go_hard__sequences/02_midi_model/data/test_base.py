import unittest

from base import Base

class TestBase(unittest.TestCase):
    def test_load(self):
        base = Base("../../stubs/chopin")

        # Check if the data is loaded correctly
        self.assertIsNotNone(base.data)

        self.assertEqual(85355, len(base.data))
        self.assertIn("step", base.data.columns)
        self.assertIn("duration", base.data.columns)
        self.assertIn("velocity", base.data.columns)
        self.assertIn("pitch", base.data.columns)
        self.assertIn("midi_idx", base.data.columns)
        self.assertIn("instrument_idx", base.data.columns)
