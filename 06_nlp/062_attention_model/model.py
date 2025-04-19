import torch


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, vocab_size, num_heads = 2, dropout=.2):
        super(MultiHeadAttention, self).__init__()
        embed_dim = vocab_size
        if embed_dim % num_heads != 0:
            embed_dim = (embed_dim // num_heads) * num_heads
        self.E = torch.nn.Embedding(vocab_size, embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(vocab_size).expand((1, -1))  # Support sequences up to length 32
        )
        self.position_embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.attn = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embed_dim, vocab_size)
        )


    def forward(self, x):
        seq_length = x.size(1)

        x = self.E(x)
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        x = self.dropout(x)
        x, _ = self.attn(x, x, x)
        x = x[:, -1, :]  # Take the last token from the sequence (if sequence classification)
        x = self.classifier(x)
        return x