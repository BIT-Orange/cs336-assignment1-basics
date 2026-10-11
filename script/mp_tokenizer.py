"""Encode the TinyStories corpora into uint16 token arrays with the trained tokenizer.

The corpus is cut into byte ranges at `<|endoftext|>` occurrences, the only safe place to
split (no pre-token ever crosses one).  Each range is encoded by its own worker process
into a part file, and the parts are concatenated in order.
"""

import codecs
import math
import multiprocessing as mp
import os
import shutil
import time
from itertools import islice
from pathlib import Path

import numpy as np

from cs336_basics.bpe import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer

TOKENIZER_DIR = Path("data/bpe/tinystories")
SPECIAL_TOKENS = ["<|endoftext|>"]
JOBS = [
    ("data/TinyStoriesV2-GPT4-train.txt", "data/tokenized/tinystories_train_uint16.bin"),
    ("data/TinyStoriesV2-GPT4-valid.txt", "data/tokenized/tinystories_valid_uint16.bin"),
]
NUM_WORKERS = min(os.cpu_count() or 1, 8)
MIN_CHUNK_BYTES = 64 * 2**20  # below this, process startup costs more than the parallel speedup
BLOCK_BYTES = 1 << 20
BATCH_TOKENS = 1 << 20


def text_blocks(path: str, start: int, end: int):
    """Decode bytes [start, end) block by block, so a character on a block edge survives."""
    decoder = codecs.getincrementaldecoder("utf-8")("ignore")
    with open(path, "rb") as source:
        source.seek(start)
        remaining = end - start
        while remaining > 0:
            block = source.read(min(BLOCK_BYTES, remaining))
            if not block:
                break
            remaining -= len(block)
            if text := decoder.decode(block):
                yield text
        if tail := decoder.decode(b"", final=True):
            yield tail


def encode_range(job: tuple) -> int:
    """Encode one byte range into its own part file; runs in a worker process."""
    input_path, start, end, part_path = job
    tokenizer = Tokenizer.from_files(TOKENIZER_DIR / "vocab.json", TOKENIZER_DIR / "merges.txt", SPECIAL_TOKENS)
    tokens = tokenizer.encode_iterable(text_blocks(input_path, start, end))  # one generator per range
    count = 0
    with open(part_path, "wb") as sink:
        while batch := list(islice(tokens, BATCH_TOKENS)):
            np.array(batch, dtype=np.uint16).tofile(sink)
            count += len(batch)
    return count


def encode_file(input_path: Path, output_path: Path) -> None:
    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    size = input_path.stat().st_size
    chunks = min(NUM_WORKERS, max(1, math.ceil(size / MIN_CHUNK_BYTES)))
    if SPECIAL_TOKENS and chunks > 1:
        with input_path.open("rb") as file:
            boundaries = find_chunk_boundaries(file, chunks, SPECIAL_TOKENS[0].encode("utf-8"))
    else:
        boundaries = [0, size]
    ranges = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:]) if start < end]

    parts = [output_path.parent / f"{output_path.name}.part{index}" for index in range(len(ranges))]
    jobs = [(str(input_path), start, end, str(part)) for (start, end), part in zip(ranges, parts)]
    if len(jobs) == 1:
        counts = [encode_range(jobs[0])]
    else:
        with mp.Pool(len(jobs)) as pool:
            counts = pool.map(encode_range, jobs)

    with output_path.open("wb") as sink:
        for part in parts:
            with part.open("rb") as piece:
                shutil.copyfileobj(piece, sink, length=1 << 22)
            part.unlink()
    print(f"{output_path}: {sum(counts):,} tokens, {len(ranges)} chunk(s), {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    for source, destination in JOBS:
        encode_file(Path(source), Path(destination))
