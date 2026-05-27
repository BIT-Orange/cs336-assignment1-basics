from __future__ import annotations

import heapq
import math
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

import regex as re


GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PRETOKENIZER_PATTERN = re.compile(GPT2_PRETOKENIZER_PATTERN)
_BYTE_TOKENS = tuple(bytes([byte]) for byte in range(256))


Token = tuple[bytes, ...]
Pair = tuple[bytes, bytes]
CountProgressCallback = Callable[[int, int, int, int], None]
MergeProgressCallback = Callable[[int, int, Pair, int], None]


class _ReversePair:
    __slots__ = ("pair",)

    def __init__(self, pair: Pair) -> None:
        self.pair = pair

    def __lt__(self, other: _ReversePair) -> bool:
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ReversePair) and self.pair == other.pair


PairHeap = list[tuple[int, _ReversePair, Pair]]


def _initial_vocab(special_tokens: Iterable[str]) -> dict[int, bytes]:
    vocab = {i: _BYTE_TOKENS[i] for i in range(256)}
    next_id = 256
    for special_token in special_tokens:
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1
    return vocab


def _special_token_split_pattern(special_tokens: list[str]) -> re.Pattern[str] | None:
    if not special_tokens:
        return None
    return re.compile("|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))


def _non_special_segments(text: str, split_pattern: re.Pattern[str] | None) -> Iterator[str]:
    if split_pattern is None:
        yield text
        return

    start = 0
    for match in split_pattern.finditer(text):
        if start < match.start():
            yield text[start : match.start()]
        start = match.end()
    if start < len(text):
        yield text[start:]


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[Token]:
    split_pattern = _special_token_split_pattern(special_tokens)

    counts: Counter[Token] = Counter()
    for segment in _non_special_segments(text, split_pattern):
        for match in _PRETOKENIZER_PATTERN.finditer(segment):
            pretoken_bytes = match.group().encode("utf-8")
            counts[tuple(_BYTE_TOKENS[byte] for byte in pretoken_bytes)] += 1
    return counts


def _pair_frequencies(token: Token) -> Counter[Pair]:
    return Counter(zip(token, token[1:]))


def _add_token_pairs(
    token: Token,
    count: int,
    pair_counts: Counter[Pair],
    pair_to_tokens: dict[Pair, set[Token]],
    pair_heap: PairHeap | None = None,
) -> None:
    for pair, occurrences in _pair_frequencies(token).items():
        pair_counts[pair] += count * occurrences
        pair_to_tokens[pair].add(token)
        if pair_heap is not None:
            heapq.heappush(pair_heap, (-pair_counts[pair], _ReversePair(pair), pair))


def _remove_token_pairs(
    token: Token,
    count: int,
    pair_counts: Counter[Pair],
    pair_to_tokens: dict[Pair, set[Token]],
    pair_heap: PairHeap | None = None,
) -> None:
    for pair, occurrences in _pair_frequencies(token).items():
        new_count = pair_counts[pair] - count * occurrences
        if new_count <= 0:
            pair_counts.pop(pair, None)
            pair_to_tokens.pop(pair, None)
        else:
            pair_counts[pair] = new_count
            pair_to_tokens[pair].discard(token)
            if not pair_to_tokens[pair]:
                pair_to_tokens.pop(pair, None)
            if pair_heap is not None:
                heapq.heappush(pair_heap, (-new_count, _ReversePair(pair), pair))


def _merge_pair_in_token(token: Token, pair_to_merge: Pair) -> Token:
    merged_token = pair_to_merge[0] + pair_to_merge[1]
    result: list[bytes] = []
    i = 0
    while i < len(token):
        if i + 1 < len(token) and token[i] == pair_to_merge[0] and token[i + 1] == pair_to_merge[1]:
            result.append(merged_token)
            i += 2
        else:
            result.append(token[i])
            i += 1
    return tuple(result)


def _pop_best_pair(pair_counts: Counter[Pair], pair_heap: PairHeap) -> Pair | None:
    while pair_heap:
        neg_count, _, pair = heapq.heappop(pair_heap)
        if pair_counts.get(pair, 0) == -neg_count:
            return pair
    return None


def _train_bpe_from_token_counts(
    token_counts: Counter[Token],
    vocab_size: int,
    special_tokens: list[str],
    *,
    merge_progress_callback: MergeProgressCallback | None = None,
    merge_progress_every: int = 1000,
) -> tuple[dict[int, bytes], list[Pair]]:
    vocab = _initial_vocab(special_tokens)
    num_merges_to_train = vocab_size - len(vocab)
    if vocab_size <= len(vocab):
        return {idx: vocab[idx] for idx in range(vocab_size)}, []

    pair_counts: Counter[Pair] = Counter()
    pair_to_tokens: dict[Pair, set[Token]] = defaultdict(set)
    for token, count in token_counts.items():
        _add_token_pairs(token, count, pair_counts, pair_to_tokens)
    pair_heap: PairHeap = [(-count, _ReversePair(pair), pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)

    merges: list[Pair] = []
    while len(vocab) < vocab_size and pair_heap:
        best_pair = _pop_best_pair(pair_counts, pair_heap)
        if best_pair is None:
            break
        affected_tokens = list(pair_to_tokens.get(best_pair, ()))
        if not affected_tokens:
            pair_counts.pop(best_pair, None)
            continue

        best_pair_count = pair_counts[best_pair]
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        if merge_progress_callback is not None and (
            len(merges) == 1 or len(merges) % merge_progress_every == 0 or len(vocab) == vocab_size
        ):
            merge_progress_callback(len(merges), num_merges_to_train, best_pair, best_pair_count)

        for old_token in affected_tokens:
            count = token_counts.pop(old_token, 0)
            if count == 0:
                continue

            _remove_token_pairs(old_token, count, pair_counts, pair_to_tokens, pair_heap)
            new_token = _merge_pair_in_token(old_token, best_pair)
            token_counts[new_token] += count
            _add_token_pairs(new_token, count, pair_counts, pair_to_tokens, pair_heap)

    return vocab, merges


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    **_: object,
) -> tuple[dict[int, bytes], list[Pair]]:
    special_tokens = special_tokens or []
    text = Path(input_path).read_text(encoding="utf-8")
    token_counts = _pretoken_counts(text, special_tokens)
    return _train_bpe_from_token_counts(token_counts, vocab_size, special_tokens)


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
    *,
    max_end: int | None = None,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    end_position = file_size if max_end is None else min(max_end, file_size)
    file.seek(0)

    if desired_num_chunks <= 1 or end_position == 0:
        return [0, end_position]

    chunk_size = max(1, end_position // desired_num_chunks)
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = end_position

    mini_chunk_size = 4096
    for boundary_index in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[boundary_index]
        file.seek(initial_position)

        while initial_position < end_position:
            bytes_to_read = min(mini_chunk_size, end_position - initial_position)
            mini_chunk = file.read(bytes_to_read)
            if mini_chunk == b"":
                chunk_boundaries[boundary_index] = end_position
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[boundary_index] = initial_position + found_at
                break
            initial_position += len(mini_chunk)
        else:
            chunk_boundaries[boundary_index] = end_position

    return sorted(set(chunk_boundaries))


def _count_pretokens_in_range(args: tuple[str, int, int, tuple[str, ...]]) -> tuple[Counter[Token], int]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8", errors="ignore")
    return _pretoken_counts(text, list(special_tokens)), end - start


def count_pretokens_from_file(
    input_path: str | Path,
    special_tokens: list[str] | None = None,
    *,
    num_workers: int = 1,
    chunk_size_mb: int = 256,
    split_special_token: str | None = None,
    max_bytes: int | None = None,
    progress_callback: CountProgressCallback | None = None,
) -> Counter[Token]:
    special_tokens = special_tokens or []
    input_path = Path(input_path)
    if chunk_size_mb <= 0:
        raise ValueError("chunk_size_mb must be positive")

    if split_special_token is None:
        split_special_token = special_tokens[0] if special_tokens else None

    if split_special_token is None:
        if max_bytes is None:
            return _pretoken_counts(input_path.read_text(encoding="utf-8"), special_tokens)
        with input_path.open("rb") as file:
            text = file.read(max_bytes).decode("utf-8", errors="ignore")
        return _pretoken_counts(text, special_tokens)

    file_size = input_path.stat().st_size
    effective_size = file_size if max_bytes is None else min(file_size, max_bytes)
    target_chunk_bytes = chunk_size_mb * 1024 * 1024
    desired_num_chunks = max(1, math.ceil(effective_size / target_chunk_bytes))

    with input_path.open("rb") as file:
        boundaries = _find_chunk_boundaries(
            file,
            desired_num_chunks,
            split_special_token.encode("utf-8"),
            max_end=effective_size,
        )

    jobs = [
        (str(input_path), start, end, tuple(special_tokens))
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]
    token_counts: Counter[Token] = Counter()
    if not jobs:
        return token_counts

    completed_chunks = 0
    processed_bytes = 0
    worker_count = max(1, min(num_workers, len(jobs)))

    if worker_count == 1:
        for job in jobs:
            chunk_counts, bytes_read = _count_pretokens_in_range(job)
            token_counts.update(chunk_counts)
            completed_chunks += 1
            processed_bytes += bytes_read
            if progress_callback is not None:
                progress_callback(completed_chunks, len(jobs), processed_bytes, len(token_counts))
        return token_counts

    with mp.Pool(processes=worker_count) as pool:
        for chunk_counts, bytes_read in pool.imap_unordered(_count_pretokens_in_range, jobs, chunksize=1):
            token_counts.update(chunk_counts)
            completed_chunks += 1
            processed_bytes += bytes_read
            if progress_callback is not None:
                progress_callback(completed_chunks, len(jobs), processed_bytes, len(token_counts))

    return token_counts


def train_bpe_from_file(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    *,
    num_workers: int = 1,
    chunk_size_mb: int = 256,
    split_special_token: str | None = None,
    max_bytes: int | None = None,
    count_progress_callback: CountProgressCallback | None = None,
    merge_progress_callback: MergeProgressCallback | None = None,
    merge_progress_every: int = 1000,
) -> tuple[dict[int, bytes], list[Pair]]:
    special_tokens = special_tokens or []
    token_counts = count_pretokens_from_file(
        input_path,
        special_tokens,
        num_workers=num_workers,
        chunk_size_mb=chunk_size_mb,
        split_special_token=split_special_token,
        max_bytes=max_bytes,
        progress_callback=count_progress_callback,
    )
    return _train_bpe_from_token_counts(
        token_counts,
        vocab_size,
        special_tokens,
        merge_progress_callback=merge_progress_callback,
        merge_progress_every=merge_progress_every,
    )
