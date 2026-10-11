import math
import torch

from collections.abc import Callable, Iterable


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                
                state = self.state[param]
                alpha = group["lr"]
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                t = state.get("t", 1)
                
                alpha_t = alpha * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                param.data -= alpha * group["weight_decay"] * param.data
                
                prev_m_t = state.get("m", torch.zeros_like(param.data))
                prev_v_t = state.get("v", torch.zeros_like(param.data))
                
                m_t = beta_1 * prev_m_t + (1 - beta_1) * grad
                v_t = beta_2 * prev_v_t + (1 - beta_2) * grad.pow(2)
                
                param.data -= alpha_t * m_t / (v_t.sqrt() + eps)
                
                state["m"] = m_t
                state["v"] = v_t
                state["t"] = t + 1
                
        return loss
                

def get_cosine_lr(
    iter: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    cosine_cycle_iters: int
):
    if iter < warmup_steps:
        return max_learning_rate * iter / warmup_steps
    if iter > cosine_cycle_iters:
        return min_learning_rate
    decay_ratio = (iter - warmup_steps) / (cosine_cycle_iters - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
                