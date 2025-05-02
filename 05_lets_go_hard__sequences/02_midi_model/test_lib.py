import unittest
from data.lib import build_ngrams

class TestLib(unittest.TestCase):
    def test_ngrams_2_grams(self):
        arr = [1, 2, 3, 4]
        n = 2

        expected_ngrams = [([1, 2], 3), ([2, 3], 4)]
        result = build_ngrams(arr, n)
        self.assertEqual(expected_ngrams, result, f"Expected {expected_ngrams}, but got {result}")


    def test_ngrams_3_grams(self):
        arr = [1, 2, 3, 4, 5, 6, 7]
        n = 3

        expected_ngrams = [([1, 2, 3], 4), ([2, 3, 4], 5), ([3, 4, 5], 6), ([4, 5, 6], 7)]
        result = build_ngrams(arr, n)
        self.assertEqual(expected_ngrams, result, f"Expected {expected_ngrams}, but got {result}")