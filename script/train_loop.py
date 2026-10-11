import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import (
    cross_entropy_loss,
    gradient_clip,
    load_checkpoint,
    save_checkpoint,
)
from cs336_basics.optimizer import AdamW, get_cosine_lr

# ===================== CONFIG =====================
TRAIN_DATA = "data/tokenized/tinystories_train_uint16.bin"
VALID_DATA = "data/tokenized/tinystories_valid_uint16.bin"
DATA_DTYPE = np.uint16  # must match how the file on disk was written
CHECKPOINT = "train/output/simple/checkpoint.pt"
RESUME = False  # True: continue from CHECKPOINT if it exists
DEVICE = "mps" 
SEED = 42

# model
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
ROPE_THETA = 10_000.0

# optimization
BATCH_SIZE = 32
MAX_STEPS = 5_000
LR_MAX = 3e-4
LR_MIN = 3e-5
WARMUP_STEPS = 200
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0

# logging and checkpointing
LOG_EVERY = 10
EVAL_EVERY = 200
EVAL_BATCHES = 20
CHECKPOINT_EVERY = 1_000
# ==================================================

def load_tokens(path: str) -> np.memmap:
    """Memory-map a token file, so the OS loads only the parts we touch."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found - run encode_tinystories.py first to create it.")
    return np.memmap(path, dtype=DATA_DTYPE, mode="r")


@torch.no_grad()
def estimate_loss(model: BasicsTransformerLM, data: np.memmap, device: str) -> float:
    """Average loss over EVAL_BATCHES random batches (no gradient updates)."""
    model.eval()
    total = 0.0
    for _ in range(EVAL_BATCHES):
        inputs, targets = get_batch(data, BATCH_SIZE, CONTEXT_LENGTH, device)
        total += cross_entropy_loss(model(inputs), targets).item()
    model.train()
    return total / EVAL_BATCHES


def main() -> None:
    torch.manual_seed(SEED)

    train_data = load_tokens(TRAIN_DATA)
    valid_data = load_tokens(VALID_DATA)

    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        theta=ROPE_THETA,
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LR_MAX,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )

    Path(CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
    start_step = 0
    if RESUME and Path(CHECKPOINT).is_file():
        start_step = load_checkpoint(CHECKPOINT, model, optimizer)
        print(f"resumed from step {start_step}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"device={DEVICE} parameters={num_params:,} steps={MAX_STEPS}")
    started_at = time.perf_counter()

    for step in range(start_step, MAX_STEPS):
        # 1. learning rate for this step: warmup, then cosine decay
        lr = get_cosine_lr(step, LR_MAX, LR_MIN, WARMUP_STEPS, MAX_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2. forward pass, loss, backward pass
        optimizer.zero_grad()
        inputs, targets = get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        loss = cross_entropy_loss(model(inputs), targets)
        loss.backward()

        # 3. clip gradients, then update the weights
        gradient_clip(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        # 4. log, evaluate, checkpoint
        done = step + 1
        if done % LOG_EVERY == 0:
            minutes = (time.perf_counter() - started_at) / 60
            print(f"step {done:>6}  train loss {loss.item():.4f}  lr {lr:.2e}  {minutes:.1f} min")
        if done % EVAL_EVERY == 0:
            valid_loss = estimate_loss(model, valid_data, DEVICE)
            print(f"step {done:>6}  valid loss {valid_loss:.4f}")
        if done % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, optimizer, done, CHECKPOINT)

    save_checkpoint(model, optimizer, MAX_STEPS, CHECKPOINT)
    print(f"saved checkpoint to {CHECKPOINT}")


if __name__ == "__main__":
    main()
