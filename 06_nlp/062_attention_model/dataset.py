from torch.utils.data import Dataset, Subset
import torch
from typing import List
import re

PAD_TEXT = '<PAD>'

vocab = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
         '+': 10, '-': 11, '*': 12, '=': 13, '<PAD>': 14}

decode_vocab = {out: key for key, out in vocab.items()}

def decode_tokens(tokens):
    decoded = []
    for token in tokens:
        token = token.item()
        if token == vocab[PAD_TEXT]:
            break
        decoded.append(decode_vocab[token])
    return "".join(decoded)

def train_test_split_sequential(
        dataset: Dataset,
        test_size: float = 0.2,
):
    """
    Splits a dataset into training and testing sets sequentially.
    :param dataset: The dataset to split.
    :param test_size: The proportion of the dataset to include in the test split.
    :return: train_dataset, test_dataset
    """
    n = len(dataset)
    test_len = int(n * test_size)
    train_len = n - test_len

    train_indices = list(range(train_len))
    test_indices = list(range(train_len, n))

    return Subset(dataset, train_indices), Subset(dataset, test_indices)

def tokenize_line(line, vocab):
    tokens = []
    # Use regex to split into numbers (including negatives) and operators
    elements = re.findall(r'-?\d+|[+\-*=]', line.replace(" ", ""))  # Match numbers and operators
    for element in elements:
        if element.isdigit() or (element.startswith('-') and element[1:].isdigit()):  # Handle numbers
            for char in element:  # Tokenize each digit separately
                tokens.append(vocab[char])
        elif element in vocab:  # Handle operators
            tokens.append(vocab[element])
    return tokens

def pad_tokens(tokens: List[List], pad_token=vocab[PAD_TEXT]) -> List[List]:
    max_len = max([len(tokenized) for tokenized in tokens])
    padded = []
    for tokenized in tokens:
        padded_data = tokenized + [pad_token] * (max_len - len(tokenized))
        padded.append(padded_data)

    return padded

class BasicOperationsDataset(Dataset):
    def __init__(self, file_name: str):
        self._raw_lines = []
        self._raw_operations = []
        self._raw_results = []
        with open(file_name, "r") as file:
            for line in file:
                line = line.strip()
                self._raw_lines.append(line)
                # a line is composed by (operation) = (result) -- result can be negative numbers

                operation, result = line.split("=")
                self._raw_operations.append(operation.strip() + " =")
                self._raw_results.append(result.strip())

        tokenized_operations = [tokenize_line(line, vocab) for line in self._raw_operations]
        padded_operations = pad_tokens(tokenized_operations)

        tokenized_results = [tokenize_line(line, vocab)[0] for line in self._raw_results]

        self.x = padded_operations
        self.y = torch.tensor(tokenized_results, dtype=torch.long)

    def __len__(self):
        return len(self._raw_operations)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), self.y[idx]