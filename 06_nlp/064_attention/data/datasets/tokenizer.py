import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import re
import random

class TokenizerDataset(Dataset):
    def __init__(
            self,
            file_name: str,
            batch_size_words=100,
            max_token_length=600,
            amount_of_samples=None
    ):
        self.batch_size_words = batch_size_words
        self.max_token_length = max_token_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        with open(file_name) as file:
            self._raw_content = file.read()

        # Split text into words while preserving whitespace
        self.words_with_spaces = re.findall(r'\S+|\s+', self._raw_content)
        # Create word batches and tokenize them
        self.tokenized_batches = []
        self.batch_start_indices = []

        current_token_index = 0

        # Process text in word batches (counting only non-whitespace as words)
        word_count = 0
        current_batch = []

        for item in self.words_with_spaces:
            current_batch.append(item)

            # Only count non-whitespace items as words
            if item.strip():
                word_count += 1

            # When we have enough words, tokenize the batch
            if word_count >= self.batch_size_words:
                text_batch = ''.join(current_batch)

                # Tokenize the batch
                tokens = self.tokenizer.encode(text_batch)

                # Store the starting index for this batch
                self.batch_start_indices.append(current_token_index)
                self.tokenized_batches.append(tokens)

                current_token_index += len(tokens)

                # Reset for next batch
                word_count = 0
                current_batch = []

        # Don't forget the last batch
        if current_batch:
            text_batch = ''.join(current_batch)
            tokens = self.tokenizer.encode(text_batch)
            self.batch_start_indices.append(current_token_index)
            self.tokenized_batches.append(tokens)

        # Create a single tokenized tensor for the entire text
        self.tokenized = torch.tensor([token for batch in self.tokenized_batches for token in batch])

        self._vocab_size = len(self.tokenizer)

        self._indexes = []
        for i in range(0, len(self.tokenized) - self.max_token_length, self.max_token_length):
            self._indexes.append(i)

        # If there's a remainder, include it as the last sequence
        if len(self.tokenized) - self.max_token_length > 0:
            last_valid_start = len(self.tokenized) - self.max_token_length
            if last_valid_start not in self._indexes:
                self._indexes.append(last_valid_start)

        if amount_of_samples is not None:
            if amount_of_samples < len(self._indexes):
                pass
            else:
                for i in range(len(self._indexes), amount_of_samples):
                    self._indexes.append(random.randint(0, len(self.tokenized) - self.max_token_length))

        # Shuffle the indexes
        random.shuffle(self._indexes)

        self._num_sequences = len(self._indexes)

    def __len__(self):
        return self._num_sequences

    def __getitem__(self, idx):
        start_idx = self._indexes[idx]
        end_idx = min(start_idx + self.max_token_length, len(self.tokenized))

        # Get input sequence
        x = self.tokenized[start_idx:end_idx]

        # If the sequence is shorter than max_token_length, pad it
        if len(x) < self.max_token_length:
            padding = torch.full((self.max_token_length - len(x),), self.tokenizer.pad_token_id)
            x = torch.cat([x, padding])

        # Get target sequence (shifted by 1)
        y_start = start_idx + 1
        y_end = min(y_start + self.max_token_length, len(self.tokenized))
        y = self.tokenized[y_start:y_end]

        # Pad y if necessary
        if len(y) < self.max_token_length:
            padding = torch.full((self.max_token_length - len(y),), self.tokenizer.pad_token_id)
            y = torch.cat([y, padding])

        return x, y

    @property
    def vocab_size(self):
        return self._vocab_size