from __future__ import annotations

import argparse
import cProfile
import json
import os
import pickle
import platform
import pstats
import sys
import threading
import time
from pathlib import Path

import psutil

from cs336_basics.bpe import count_pretoken_frequencies, train_bpe_from_token_counts


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop_event.is_set():
            rss_bytes = 0
            try:
                child_processes = process.children(recursive=True)
            except (psutil.Error, PermissionError):
                child_processes = []
            for current_process in (process, *child_processes):
                try:
                    rss_bytes += current_process.memory_info().rss
                except (psutil.Error, PermissionError):
                    continue
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
            self._stop_event.wait(self.interval_seconds)


def serialize_results(
    output_dir: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vocab.txt").open("w", encoding="utf-8") as file:
        for token_id, token_bytes in vocab.items():
            file.write(f"{token_id}\t{token_bytes!r}\n")

    with (output_dir / "merges.txt").open("w", encoding="utf-8") as file:
        for left, right in merges:
            file.write(f"{left!r}\t{right!r}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and profile the CS336 byte-level BPE tokenizer.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument(
        "--token-counts-cache",
        type=Path,
        help="Load pre-token counts from this pickle if it exists; otherwise compute and save them here.",
    )
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile() if args.profile else None
    memory_sampler = None if args.profile else MemorySampler()
    if memory_sampler is not None:
        memory_sampler.start()

    total_start = time.perf_counter()
    pretoken_start = time.perf_counter()
    loaded_token_counts_cache = args.token_counts_cache is not None and args.token_counts_cache.exists()
    if loaded_token_counts_cache:
        with args.token_counts_cache.open("rb") as file:
            token_counts = pickle.load(file)
        if not isinstance(token_counts, dict):
            raise ValueError(f"Token-count cache does not contain a mapping: {args.token_counts_cache}")
    else:
        token_counts = count_pretoken_frequencies(args.input_path, args.special_token, args.num_workers)
        if args.token_counts_cache is not None:
            args.token_counts_cache.parent.mkdir(parents=True, exist_ok=True)
            with args.token_counts_cache.open("wb") as file:
                pickle.dump(token_counts, file, protocol=pickle.HIGHEST_PROTOCOL)
    pretoken_seconds = time.perf_counter() - pretoken_start
    unique_pretokens = len(token_counts)

    merge_start = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    vocab, merges = train_bpe_from_token_counts(token_counts, args.vocab_size, args.special_token)
    if profiler is not None:
        profiler.disable()
    merge_seconds = time.perf_counter() - merge_start
    total_seconds = time.perf_counter() - total_start

    if memory_sampler is not None:
        memory_sampler.stop()

    serialize_results(args.output_dir, vocab, merges)
    special_token_bytes = {token.encode("utf-8") for token in args.special_token}
    longest_token_id, longest_token = max(vocab.items(), key=lambda item: (len(item[1]), item[1]))
    non_special_vocab = [item for item in vocab.items() if item[1] not in special_token_bytes]
    longest_non_special_token_id, longest_non_special_token = max(
        non_special_vocab, key=lambda item: (len(item[1]), item[1])
    )
    metrics = {
        "command": sys.argv,
        "python_version": sys.version,
        "platform": platform.platform(),
        "input_path": str(args.input_path),
        "input_size_bytes": args.input_path.stat().st_size,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "merge_count": len(merges),
        "special_tokens": args.special_token,
        "num_workers": args.num_workers,
        "token_counts_cache": str(args.token_counts_cache) if args.token_counts_cache is not None else None,
        "loaded_token_counts_cache": loaded_token_counts_cache,
        "unique_pretokens": unique_pretokens,
        "pretoken_seconds": pretoken_seconds,
        "merge_seconds": merge_seconds,
        "total_seconds": total_seconds,
        "sampled_peak_rss_bytes": memory_sampler.peak_rss_bytes if memory_sampler is not None else None,
        "longest_token_id": longest_token_id,
        "longest_token_length_bytes": len(longest_token),
        "longest_token_repr": repr(longest_token),
        "longest_non_special_token_id": longest_non_special_token_id,
        "longest_non_special_token_length_bytes": len(longest_non_special_token),
        "longest_non_special_token_repr": repr(longest_non_special_token),
        "profile_enabled": args.profile,
        "profile_scope": "merge_phase" if args.profile else None,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
        file.write("\n")

    if profiler is not None:
        profiler.dump_stats(args.output_dir / "profile.prof")
        with (args.output_dir / "profile.txt").open("w", encoding="utf-8") as file:
            stats = pstats.Stats(profiler, stream=file)
            stats.strip_dirs().sort_stats("cumulative").print_stats(50)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
