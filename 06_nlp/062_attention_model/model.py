import torch


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, vocab_size, num_heads = 2):
        super(MultiHeadAttention, self).__init__()
        self.E = torch.nn.Embedding(vocab_size, vocab_size)
        self.attn = torch.nn.MultiheadAttention(embed_dim=vocab_size, num_heads=num_heads, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(vocab_size, vocab_size),
            torch.nn.ReLU(),
            torch.nn.Linear(vocab_size, vocab_size)
        )


    def forward(self, x):
        x = self.E(x)
        x, _ = self.attn(x, x, x)
        x = x[:, -1, :]  # Take the last token from the sequence (if sequence classification)
        x = self.classifier(x)
        return x