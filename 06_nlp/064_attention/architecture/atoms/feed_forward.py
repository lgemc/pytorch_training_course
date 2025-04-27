from torch import nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim) -> None:
        super().__init__()
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x): # x: [batch_size, block_size, head_dim]
        dense1 = F.relu(self.dense1(x)) # [batch_size, block_size, head_dim*4]
        output = self.dense2(dense1) # [batch_size, block_size, head_dim]
        return output