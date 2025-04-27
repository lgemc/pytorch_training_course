import unittest
import torch

from models.from_scratch import Basic
from data.datasets.tokenizer import  TokenizerDataset
from generate import generate_text


block_size = 140
model_dim = 64
heads_num = 4
blocks_num = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

class TestGenerate(unittest.TestCase):
    def test_generate(self):
        dataset = TokenizerDataset(
            "../data/static/input.txt",
                    batch_size_words=140,
                    max_token_length=140,
        )

        model = Basic(
            vocab_size=dataset._vocab_size,
            block_size=block_size,
            model_dim=model_dim,
            heads_num=heads_num,
            blocks_num=blocks_num,
            device=device
        )

        model.load_state_dict(torch.load("model.pth"))

        prompt = "ARNALDO:"

        generated =  generate_text(
            model=model,
            tokenizer=dataset.tokenizer,
            prompt=prompt,
            max_new_tokens=250,
            temperature=0.8,
            top_p=0.95,
            device=device
        )

        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")