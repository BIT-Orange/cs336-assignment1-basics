from __future__ import annotations

import heapq
import math
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import BinaryIO

import regex as re

GPT2_PRETOKENIZER = re.compile(r"""'(?:[smdt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
BYTE_TOKENS = tuple(bytes([byte]) for byte in range(256))

Token = tuple[bytes, ...]
Pair = tuple[bytes, bytes]


class ReversePair:
    __slots__ = ("pair",)

    def __init__(self, pair: Pair):
        self.pair = pair

    def __lt__(self, other: ReversePair) -> bool:
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReversePair) and self.pair == other.pair


PairHeap = list[tuple[int, ReversePair, Pair]]


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
                counts[tuple(BYTE_TOKENS[b] for b in pretoken_bytes)] += 1
    return counts


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


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
    if not jobs:
        pass
    elif num_workers <= 1 or len(jobs) == 1:
        for job in jobs:
            token_counts.update(chunk_worker(*job))
    else:
        worker_count = min(num_workers, len(jobs))
        with mp.Pool(processes=worker_count) as pool:
            for result in pool.imap_unordered(_chunk_worker_wrapper, jobs):
                token_counts.update(result)

    return token_counts


def train_bpe_from_token_counts(
    token_counts: Counter[Token],
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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

    pair_counts: Counter[Pair] = Counter()
    pair_to_tokens: dict[Pair, set[Token]] = defaultdict(set)

    for token, count in token_counts.items():
        for pair, occurrences in Counter(zip(token, token[1:])).items():
            pair_counts[pair] += count * occurrences
            pair_to_tokens[pair].add(token)

    pair_heap: PairHeap = [(-count, ReversePair(pair), pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)

    merges: list[Pair] = []
    while len(vocab) < vocab_size and pair_heap:
        best_pair: Pair | None = None
        while pair_heap:
            neg, _, pair = heapq.heappop(pair_heap)
            if -neg == pair_counts.get(pair, 0):
                best_pair = pair
                break
        if best_pair is None:
            break
        affect_tokens = list(pair_to_tokens.get(best_pair, ()))
        if not affect_tokens:
            pair_counts.pop(best_pair, None)
            pair_to_tokens.pop(best_pair, None)
            continue

        merges.append(best_pair)
        merged_bytes = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_bytes

        new_token_counts: Counter[Token] = Counter()
        for token in affect_tokens:
            count = token_counts.pop(token, 0)
            if count == 0:
                continue

            for pair, occurrences in Counter(zip(token, token[1:])).items():
                pair_counts[pair] -= count * occurrences
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                    pair_to_tokens.pop(pair, None)
                else:
                    pair_to_tokens[pair].discard(token)
                    if not pair_to_tokens[pair]:
                        pair_to_tokens.pop(pair, None)
                    heapq.heappush(pair_heap, (-pair_counts[pair], ReversePair(pair), pair))

            result: list[bytes] = []
            i = 0
            while i < len(token):
                if i + 1 < len(token) and token[i : i + 2] == best_pair:
                    result.append(merged_bytes)
                    i += 2
                else:
                    result.append(token[i])
                    i += 1
            new_token = tuple(result)
            new_token_counts[new_token] += count

        for new_token, count in new_token_counts.items():
            token_counts[new_token] += count
            for pair, occurrences in Counter(zip(new_token, new_token[1:])).items():
                pair_counts[pair] += count * occurrences
                pair_to_tokens[pair].add(new_token)
                heapq.heappush(pair_heap, (-pair_counts[pair], ReversePair(pair), pair))
    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_workers: int | None = None,
    **_: object,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    normalized_special_tokens = list(dict.fromkeys(special_tokens or []))
    worker_count = num_workers if num_workers is not None else min(os.cpu_count() or 1, 8)
    token_counts = count_pretoken_frequencies(input_path, normalized_special_tokens, worker_count)
    return train_bpe_from_token_counts(token_counts, vocab_size, normalized_special_tokens)
