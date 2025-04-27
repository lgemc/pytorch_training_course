import torch

def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    soft = torch.nn.functional.log_softmax(logits, dim=1)
    real_values = soft[range(len(y)), y]
    return -real_values.mean()