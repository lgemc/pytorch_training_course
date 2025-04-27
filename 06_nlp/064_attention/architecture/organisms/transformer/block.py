import torch.nn as nn
from architecture.molecules.multihead_attention import MultiHeadAttention
from architecture.atoms.feed_forward import FeedForward

class Block(nn.Module):
    def __init__(self, heads_num, model_dim, block_size, drop_out=.3) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(heads_num, model_dim, model_dim, block_size)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ffd = FeedForward(model_dim, model_dim*4, model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

        self.drop1 = nn.Dropout(drop_out)
        self.drop2 = nn.Dropout(drop_out)
        self.drop3 = nn.Dropout(drop_out)

    def forward(self, x): # x: [batch_size, block_size, emb_dim]
        att = self.attention(x) # [batch_size, block_size, head_dim], emb_dim y head_dim deben ser iguales para que funcionen las conexiones residuales
        att = self.drop1(att)
        x = self.ln1(att + x)

        ffd = self.ffd(x) # [batch_size, block_size, head_dim]
        ffd = self.drop2(ffd)
        x = self.ln2(ffd + x)
        x = self.drop3(x)
        return x # [batch_size, block_size, head_dim]