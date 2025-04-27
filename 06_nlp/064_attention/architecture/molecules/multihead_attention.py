import torch
from torch import nn

from architecture.atoms.self_attention import  SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, heads_num, embed_dim, head_dim, block_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(embed_dim, head_dim//heads_num, block_size) for _ in range(heads_num)])
        self.dense = nn.Linear(head_dim, head_dim, bias=False)

    def forward(self, x): # x: [batch_size, block_size, emb_dim]
        heads = [h(x) for h in self.heads] # [batch_size, block_size, head_dim/heads_num] por cada elemento en la lista
        att = torch.concat(heads, dim=-1) # Se concatenan los resultados de cada cabeza para obtener [batch_size, block_size, head_dim]
        output = self.dense(att) # [batch_size, block_size, head_dim]
        return output