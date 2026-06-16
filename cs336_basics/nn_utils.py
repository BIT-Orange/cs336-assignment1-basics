import os
from typing import Any, BinaryIO, IO
import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    
    x -= x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x) * x

def cross_entropy_loss(inputs, targets):
    # z = x - x_max
    z = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    # log_softmax(x_i) = z_i - log(sum(exp(z_j)))
    log_probs = z - torch.log(torch.sum(torch.exp(z), dim=-1, keepdim=True))
    loss = - log_probs.gather(dim=-1, index=targets.unsqueeze(-1))
    return loss.mean()

def gradient_clip(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = torch.tensor(0.0, device=grads[0].device)
    for grad in grads:
        norm += (grad** 2).sum()
    norm = torch.sqrt(norm)
    clip_coef = min(max_norm / (norm + 1e-6), 1.0)
    
    for grad in grads:
        grad *= clip_coef
        
def save_checkpoint(model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int, 
    out: str | os.PathLike | BinaryIO | IO[bytes]):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

