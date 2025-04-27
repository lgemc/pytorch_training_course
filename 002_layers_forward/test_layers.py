import unittest

import torch

class TestForward(unittest.TestCase):
    def  test_attention(self):
        torch.no_grad()
        input = torch.tensor([
           [1.0, 2.0],
           [3.0, 4.0]
        ])

        layer = torch.nn.MultiheadAttention(embed_dim=2, num_heads=1)
        attn_out, weights = layer.forward(input, input, input)
        print(attn_out)
        print(weights)

    def test_embeddings(self):
        tokenized = torch.tensor([
            [0, 2, 3],
            [2, 3, 4],
            [4, 5, 6]])
        embedding_dim = 2
        # vocab size is 7
        layer = torch.nn.Embedding(7, embedding_dim)
        out = layer(tokenized)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0][0]), embedding_dim)