from torch.utils.data import Dataset
import torch
from typing import List
import re

PAD_TEXT = '<PAD>'

vocab = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
         '+': 10, '-': 11, '*': 12, '=': 13, '<PAD>': 14}

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
        self._raw_operations = []
        self._tokenized_operations = []
        with open(file_name, "r") as file:
            for line in file:
                line = line.strip()
                self._raw_operations.append(line)
                self._tokenized_operations.append(tokenize_line(line, vocab))

        self.data = pad_tokens(self._tokenized_operations)

    def __len__(self):
        return len(self._raw_operations)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])