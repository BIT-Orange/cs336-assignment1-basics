from pathlib import Path

from cs336_basics.bpe import train_bpe
from cs336_basics.serialization import save_vocab_and_merges


def main():

    INPUT_PATH = Path("data/owt_train.txt")
    # INPUT_PATH = Path("data/owt_valid.txt")
    OUTPUT_DIR = Path("data/bpe/owt")
    VOCAB_SIZE = 32_000
    SPECIAL_TOKENS = ["<|endoftext|>"]

    vocab, merges = train_bpe(INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_vocab_and_merges(vocab, merges, OUTPUT_DIR / "vocab.json", OUTPUT_DIR / "merges.txt")


if __name__ == "__main__":
    main()