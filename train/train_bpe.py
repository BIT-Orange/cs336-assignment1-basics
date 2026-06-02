import pickle
import time
import os
from pathlib import Path
from cs336_basics.train_bpe import count_pretoken_frequencies, train_bpe_from_token_counts

def main():

    file_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    num_workers = min(8, os.cpu_count() or 4)

    output_dir = Path("train/output/owt")
    output_dir.mkdir(parents=True, exist_ok=True)

    token_counts_path = output_dir / "token_counts.pkl"

    print(f"Train_file: {file_path}")
    print(f"Target vocab size: {vocab_size}")
    print(f"special tokens: {special_tokens}")

    start_time = time.time()

    token_counts = count_pretoken_frequencies(file_path, special_tokens, num_workers)

    elapsed = time.time() - start_time
    
    print(f"Finished pretoken counting, spent: {elapsed:.2f}s")
    print(f"unique pretokens: {len(token_counts)}")

    with open(token_counts_path, "wb") as f:
        pickle.dump(token_counts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"token counts saved: {token_counts_path}")

    start_time = time.time()
    with open(token_counts_path, "rb") as f:
        token_counts = pickle.load(f)

    vocab, merges = train_bpe_from_token_counts(
        token_counts=token_counts,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    elapsed = time.time() - start_time
    print(f"Finished BPE merge training, spent: {elapsed:.2f}s")

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


if __name__ == "__main__":
    main()