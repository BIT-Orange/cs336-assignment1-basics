import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    
    x -= x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

