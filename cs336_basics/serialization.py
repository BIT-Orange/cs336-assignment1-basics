from __future__ import annotations

import json
from pathlib import Path

Pair = tuple[bytes, bytes]


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[Pair],
    vocab_path: str | Path,
    merges_path: str | Path,
) -> None:
    """Serialize a BPE vocabulary and merge list to disk in the canonical hex format."""
    vocab_json = {str(token_id): token.hex() for token_id, token in sorted(vocab.items())}
    Path(vocab_path).write_text(json.dumps(vocab_json) + "\n", encoding="utf-8")
    Path(merges_path).write_text(
        "".join(f"{left.hex()} {right.hex()}\n" for left, right in merges),
        encoding="utf-8",
    )


def load_vocab_and_merges(
    vocab_path: str | Path,
    merges_path: str | Path,
) -> tuple[dict[int, bytes], list[Pair]]:
    """Load a vocabulary and merge list written by save_vocab_and_merges."""
    with Path(vocab_path).open(encoding="utf-8") as file:
        vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in json.load(file).items()}

    merges: list[Pair] = []
    with Path(merges_path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                left_hex, right_hex = line.split()
                merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))
    return vocab, merges
