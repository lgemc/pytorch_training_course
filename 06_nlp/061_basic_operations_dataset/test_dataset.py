import unittest
from dataset import tokenize_line, vocab, pad_tokens, PAD_TEXT, BasicOperationsDataset

class DatasetTest(unittest.TestCase):
    def test_tokenizer(self):
        result = tokenize_line("4 + 4 = -9", vocab)
        print(result)

    def test_pad_tokens(self):
        result_1 = tokenize_line("4 - 5 = -1", vocab)
        result_2 = tokenize_line("4 + 4 = 8", vocab)

        results = [result_1, result_2]

        padded = pad_tokens(results)
        self.assertEqual(len(padded), len(results))
        self.assertEqual(len(padded[0]), 6)
        self.assertEqual(len(padded[1]), 6)
        self.assertEqual(padded[1][-1], vocab[PAD_TEXT])

    def test_dataset(self):
        dataset = BasicOperationsDataset("sample_dataset.txt")

        self.assertEqual(len(dataset), 4)
        max_len = max([len(item) for item in dataset.data])
        self.assertEqual(max_len, 6)
        self.assertEqual(dataset[1][-1], vocab[PAD_TEXT])