import json
import time
from pathlib import Path
from cs336_basics.train_bpe import train_bpe

file_path = "data/TinyStoriesV2-GPT4-valid.txt"
vocab_size = 1000
special_tokens = ["<|endoftext|>"]
output_dir = Path("train/output")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Train_file: {file_path}")
print(f"Target vocab size: {vocab_size}")
print(f"special tokens: {special_tokens}")

start_time = time.time()

vocab, merges = train_bpe(
    input_path=file_path,
    vocab_size=vocab_size,
    special_tokens=special_tokens,
    num_workers=4,
)

elapsed = time.time() - start_time
print(f"Finish training, spent: {elapsed:.2f}s")
print(f"vocab size: {len(vocab)}, merges: {len(merges)}")

vocab_path = output_dir / "vocab.txt"
with open(vocab_path, "w", encoding="utf-8") as f:
    for k, v in vocab.items():
        f.write(f"{k}\t{repr(v)}\n")
print(f"vocab saved: {vocab_path}")


merges_path = output_dir / "merges.txt"
with open(merges_path, "w", encoding="utf-8") as f:
    for token1, token2 in merges:
        f.write(f"{repr(token1)}\t{repr(token2)}\n")
print(f"merges saved: {merges_path}")

