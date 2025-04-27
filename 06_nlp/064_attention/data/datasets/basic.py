import string
import re
from typing import List

import torch
from torch.utils.data import Dataset

unknown_tok = "<unk>"

class Dataset(Dataset):
    """Basic dataset for NLP tasks.

    This dataset is used to load and preprocess the data for NLP tasks.
    It uses a simple tokenizer to split the text into tokens and creates a vocabulary.

    Args:
        file_name (str): Path to the text file.
        sequence_len (int): Length of the sequences to be generated. Default is 600.

    Attributes:
        tokenized (List): List of tokens in the text.
        _vocab_size (int): Size of the vocabulary.
        _stoi (dict): Dictionary to convert tokens to indices.
        _itos (dict): Dictionary to convert indices to tokens.

    Methods:
        __len__(): Returns the length of the dataset.
        __get_item__(idx): Returns a tuple of input and target sequences for the given index.
            x is the input sequence and y is the target sequence, y is x shifted by one in the sequence.
    """
    def __init__(self, file_name: str, sequence_len=600):
        self._sequence_len = sequence_len
        with open(file_name) as file:
            self._raw_content = file.read()

        patt = rf"\n|\s?\w+|\d+|\s|'t|'ve|'ll|'s|'d|[{string.punctuation}]+"

        tokenized = re.findall(patt, self._raw_content)

        self.tokenized = tokenized


        vocabulary = sorted(set(tokenized))
        vocabulary = [unknown_tok] + vocabulary

        self._vocab_size = len(vocabulary)
        # Diccionario para codificar los tokens
        self._stoi = {s: i for i, s in enumerate(vocabulary)}
        # Diccionario para decodificar los tokens
        self._itos = {i: s for i, s in enumerate(vocabulary)}

    def _encode_sequence(self, tokens: List) -> List:
        encoded = []
        for token in tokens:
            if token in self._stoi:
                encoded.append(self._stoi[token])
            else:
               print(f"Token {token} not in vocabulary. Skipping.")
        return encoded

    def _decode_sequence(self, tokens: List) -> List:
        decoded = []
        for token in tokens:
            if token in self._itos:
                decoded.append(self._itos[token])
            else:
                print(f"Token {token} not in vocabulary. Skipping.")
        return decoded


    def __getitem__(self, idx: int):
        lower_bound = max(0, idx - self._sequence_len)
        upper_bound = min(len(self.tokenized), idx + self._sequence_len)

        # Se obtiene el contexto de la secuencia
        context = self.tokenized[lower_bound:upper_bound]
        # Se codifica la secuencia
        encoded = self._encode_sequence(context)
        # Se obtiene la secuencia de entrada
        input_seq = encoded[:self._sequence_len]
        # Se obtiene la secuencia de salida
        target_seq = encoded[1:self._sequence_len + 1]
        # Se convierte a tensor
        x = torch.tensor(input_seq)
        y = torch.tensor(target_seq)

        # pad sequences with unknown token
        if len(x) < self._sequence_len:
            x = torch.cat([x, torch.tensor([self._stoi[unknown_tok]] * (self._sequence_len - len(x)))])
        if len(y) < self._sequence_len:
            y = torch.cat([y, torch.tensor([self._stoi[unknown_tok]] * (self._sequence_len - len(y)))])

        # Se devuelve la secuencia de entrada y la secuencia de salida
        return x, y

    def __len__(self):
        return len(self.tokenized)

