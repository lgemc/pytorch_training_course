import unittest

import torch
from transformers import GPT2Config

from data.datasets.tokenized import TokenizedDataset
from models.fine_tune import GPTShakespeare
from finetuned import train
from generate_finetune import generate_text
device = "cuda" if torch.cuda.is_available() else "cpu"


class TestFineTuned(unittest.TestCase):
    def test_init(self):
        train_set = TokenizedDataset(
            "../data/static/train.txt",
            batch_size_words=140,
            max_token_length=140,
            amount_of_samples=3000,
        )

        validation_set = TokenizedDataset(
            "../data/static/test.txt",
            batch_size_words=140,
            max_token_length=140,
            amount_of_samples=600,
        )

        model = GPTShakespeare(
            GPT2Config(
                n_embd=140,
                n_head=4,
                n_layer=4,
                attn_pdrop=.7,
                embd_pdrop=.7,
                resid_pdrop=.7,
                summary_first_dropout=0.7
            ),
            vocab_size=train_set._vocab_size,
        )

        print(model)

        model, train_losses, test_losses = train(
            model,
            train_dataset=train_set,
            val_dataset=validation_set,
            epochs=30,
            batch_size=32,
            learning_rate=0.00001,
            device=device,
            logging_steps=4,
            vocab_size=train_set._vocab_size,
        )

        torch.save(model.state_dict(), "../../../05_lets_go_hard__sequences/model_finetuned.pth")

    def test_generate(self):
        dataset = TokenizedDataset(
            "../data/static/train.txt",
            batch_size_words=140,
            max_token_length=140,
        )

        model = GPTShakespeare(
            GPT2Config(
                n_embd=140,
                n_head=4,
                n_layer=4,
            ),
            vocab_size=dataset._vocab_size,
        )

        model.load_state_dict(torch.load("../../../05_lets_go_hard__sequences/model_finetuned.pth"))

        prompt = "\n"

        generated = generate_text(
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