import math
import multiprocessing as mp
import regex as re

import os
from collections import Counter
from pathlib import Path

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.ArrayBpeTrainer import ArrayBpeTrainer

GPT2_PRETOKENIZER = re.compile(r"""'(?:[smdt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
BYTE_TOKENS = tuple(bytes([byte]) for byte in range(256))

Token = tuple[bytes, ...]
Pair = tuple[bytes, bytes]
Buckets = dict[int, list[Pair]]

    
def pretoken_count(text: str, special_tokens: list[str]) -> Counter[Token]:
    counts: Counter[Token] = Counter()

    if special_tokens:
        split_pattern = re.compile(
            "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        )
        segments = split_pattern.split(text)
    else:
        segments = [text]

    for segment in segments:
        if segment:
            for match in GPT2_PRETOKENIZER.finditer(segment):
                pretoken_bytes = match.group().encode("utf-8")
                counts[tuple(BYTE_TOKENS[byte] for byte in pretoken_bytes)] += 1
    return counts

def chunk_worker(input_path: str, start: int, end: int, special_tokens: tuple[str, ...]) -> Counter[Token]:
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
        return pretoken_count(chunk, list(special_tokens))


def _chunk_worker_wrapper(job: tuple[str, int, int, tuple[str, ...]]) -> Counter[Token]:
    return chunk_worker(*job)


def count_pretoken_frequencies(
    input_path: str | os.PathLike[str],
    special_tokens: list[str],
    num_workers: int,
) -> Counter[Token]:
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}")

    path = Path(input_path)
    file_size = path.stat().st_size
    if special_tokens and file_size >= 64 * 2**20:
        num_chunks = max(num_workers, math.ceil(file_size / (256 * 2**20)))
    else:
        num_chunks = 1

    with path.open("rb") as file:
        if not special_tokens:
            boundaries = [0, file_size]
        else:
            boundaries = find_chunk_boundaries(file, num_chunks, special_tokens[0].encode("utf-8"))

    jobs = [
        (str(path), start, end, tuple(special_tokens))
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]

    token_counts: Counter[Token] = Counter()
    if len(jobs) == 1 or num_workers <= 1:
        for job in jobs:
            token_counts.update(chunk_worker(*job))
    elif jobs:
        worker_count = min(num_workers, len(jobs))
        with mp.Pool(processes=worker_count) as pool:
            for result in pool.imap_unordered(_chunk_worker_wrapper, jobs):
                token_counts.update(result)
    return token_counts


def train_bpe_from_token_counts(
    token_counts: Counter[Token],
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[Pair]]:
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    normalized_special_tokens = list(dict.fromkeys(special_tokens or []))
    vocab = {i: BYTE_TOKENS[i] for i in range(256)}
    next_id = 256
    vocab_values = set(vocab.values())
    for token in normalized_special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab_values:
            vocab[next_id] = token_bytes
            vocab_values.add(token_bytes)
            next_id += 1

    minimum_vocab_size = len(vocab)
    if vocab_size < minimum_vocab_size:
        raise ValueError(
            "vocab_size must fit the 256 byte tokens and all unique special tokens; "
            f"minimum is {minimum_vocab_size}, got {vocab_size}"
        )
    if vocab_size == minimum_vocab_size:
        return vocab, []

    symbol_bytes = list(BYTE_TOKENS)
    symbol_map = {value: index for index, value in enumerate(symbol_bytes)}
    symbol_array_type = "H" if vocab_size <= 0x10000 else "I"
    trainer = ArrayBpeTrainer(token_counts, symbol_bytes, symbol_map, symbol_array_type)
    merges: list[Pair] = []

    while len(vocab) < vocab_size:
        pair_id = trainer.buckets.best_pair()
        if not pair_id:
            break

        left = trainer.pair_left[pair_id]
        right = trainer.pair_right[pair_id]
        left_bytes = symbol_bytes[left]
        right_bytes = symbol_bytes[right]
        merged_bytes = left_bytes + right_bytes
        merged_symbol = symbol_map.get(merged_bytes)
        if merged_symbol is None:
            merged_symbol = len(symbol_bytes)
            if merged_symbol > trainer.symbol_max:
                raise OverflowError("too many BPE symbols for the compact symbol representation")
            symbol_map[merged_bytes] = merged_symbol
            symbol_bytes.append(merged_bytes)

        merges.append((left_bytes, right_bytes))
        vocab[len(vocab)] = merged_bytes
        if not trainer.merge_pair(pair_id, merged_symbol):
            trainer.drop_pair(pair_id)

    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_workers: int | None = None,
    **_: object,
) -> tuple[dict[int, bytes], list[Pair]]:
    normalized_special_tokens = list(dict.fromkeys(special_tokens or []))
    worker_count = num_workers if num_workers is not None else min(os.cpu_count() or 1, 8)
    token_counts = count_pretoken_frequencies(input_path, normalized_special_tokens, worker_count)
    return train_bpe_from_token_counts(token_counts, vocab_size, normalized_special_tokens)

