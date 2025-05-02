import unittest

from sequences import build

class TestLib(unittest.TestCase):
    def test_build(self):
        # Test the build function with a sample input
        arr = [1, 2, 3, 4]
        n = 2

        expected_sequences = [([1, 2], [2, 3]), ([2, 3], [3, 4])]
        result = build(arr, n)
        self.assertEqual(expected_sequences, result, f"Expected {expected_sequences}, but got {result}")