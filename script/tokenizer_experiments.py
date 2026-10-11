from itertools import islice
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

TOKENIZER_DIR = Path("data/bpe/tinystories")
SPECIAL_TOKENS = ["<|endoftext|>"]
JOBS = [
    ("data/TinyStoriesV2-GPT4-train.txt", "data/tokenized/tinystories_train_uint16.bin"),
    ("data/TinyStoriesV2-GPT4-valid.txt", "data/tokenized/tinystories_valid_uint16.bin"),
]
BATCH = 1 << 20


def main():

    tokenizer = Tokenizer.from_files(TOKENIZER_DIR / "vocab.json", TOKENIZER_DIR / "merges.txt", SPECIAL_TOKENS)

    for input_path, output_path in JOBS:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, encoding="utf-8", errors="ignore") as source, open(output_path, "wb") as sink:
            tokens = tokenizer.encode_iterable(source)  # one generator per file, so the end of the text is the only cut
            while batch := list(islice(tokens, BATCH)):
                np.array(batch, dtype=np.uint16).tofile(sink)
        print(f"{output_path}: {Path(output_path).stat().st_size // 2:,} tokens")


if __name__ == "__main__":
    main()