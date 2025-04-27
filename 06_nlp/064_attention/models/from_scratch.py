import torch
from torch import nn

from architecture.organisms.transformer.block import Block

class TransformerShakespeare(nn.Module):
    def __init__(self, model_dim, vocab_size, block_size, blocks_num, heads_num, dropout=.1, device="cpu") -> None:
        super().__init__()
        self._device = device
        self.E = nn.Embedding(vocab_size, model_dim)
        self.posE = nn.Embedding(block_size, model_dim) # Embedding de posición. Cada posición en el contexto (0 - block_size-1) tiene su propio embedding
        self.ln1 = nn.LayerNorm(model_dim)
        self.blocks = nn.Sequential(*[Block(heads_num, model_dim, block_size) for _ in range(blocks_num)]) # El bloque se repite el número de veces deseado
        self.dense = nn.Linear(model_dim, vocab_size, bias=False)

        # Regularización
        self.drop1 = nn.Dropout(dropout)

    def forward(self, x): # x: [batch_size, block_size]
        emb1 = self.E(x) # [batch_size, block_size, emb_dim]

        # Positional embedding
        positions = torch.arange(x.shape[1], device=self._device) # Se genera un número por cada token que representa su posicion en el contexto (de 0 a block_size-1)
        emb2 = self.posE(positions) # [block_size, emb_dim]

        # emb1 y emb2 son de la misma forma, pero emb1 tiene batches y posE no
        emb = emb1 + emb2 # [batch_size, block_size, emb_dim] Se suman los embeddings
        emb = self.ln1(emb)
        emb = self.drop1(emb)

        x = self.blocks(emb) # [batch_size, block_size, head_dim]

        logits = self.dense(x) # [batch_size, block_size, vocab_size]
        return logits