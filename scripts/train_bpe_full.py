from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil

from cs336_basics.bpe_train import Pair, train_bpe_from_file


DATASETS = {
    "tinystories": Path("data/TinyStoriesV2-GPT4-train.txt"),
    "owt": Path("data/owt_train.txt"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on a full text corpus.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="tinystories",
        help="Convenience alias for one of the training files in data/.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Override --dataset and train on this file.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="Final vocabulary size, including byte tokens and special tokens.",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=["<|endoftext|>"],
        help="Special token to reserve and exclude from BPE merges. Repeat to add more.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes for pre-token counting.",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=64,
        help="Target chunk size for pre-token counting. Chunks are shifted to the split token.",
    )
    parser.add_argument(
        "--split-special-token",
        default="<|endoftext|>",
        help="Token used to align chunk boundaries. Use an empty string to disable boundary alignment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for vocab.json, merges.txt, and metadata.json.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Debug option: train on only the first N bytes.",
    )
    parser.add_argument(
        "--merge-progress-every",
        type=int,
        default=500,
        help="Print merge progress every N learned merges.",
    )
    return parser.parse_args()


def _write_vocab(vocab: dict[int, bytes], output_path: Path) -> None:
    output_path.write_text(
        json.dumps({str(token_id): token.hex() for token_id, token in vocab.items()}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _write_merges(merges: list[Pair], output_path: Path) -> None:
    lines = [f"{left.hex()} {right.hex()}" for left, right in merges]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _print_resource_hint(input_path: Path, num_workers: int, chunk_size_mb: int) -> None:
    available_mb = psutil.virtual_memory().available / 1024 / 1024
    file_size = input_path.stat().st_size
    approx_chunks = max(1, int(file_size / (chunk_size_mb * 1024 * 1024)) + 1)
    in_flight_mb = num_workers * chunk_size_mb

    print(
        f"count_chunks~={approx_chunks}, in_flight_raw_text~={in_flight_mb:,} MiB, "
        f"available_memory~={available_mb:,.0f} MiB",
        flush=True,
    )
    print(
        "Counting progress is printed only when a worker finishes a whole chunk; "
        "large chunks can be quiet for several minutes.",
        flush=True,
    )

    if in_flight_mb > available_mb * 0.35:
        print(
            "WARNING: this worker/chunk setting is likely memory-heavy. "
            "Try --num-workers 1 or 2 and --chunk-size-mb 64 on an 8 GiB machine.",
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    input_path = args.input_path or DATASETS[args.dataset]
    output_dir = args.output_dir or Path("data/tokenizers") / f"{input_path.stem}-vocab{args.vocab_size}"
    split_special_token = args.split_special_token or None

    if args.vocab_size < 256 + len(set(args.special_tokens)):
        raise ValueError("vocab_size must leave room for 256 byte tokens plus special tokens")
    if not input_path.exists():
        raise FileNotFoundError(f"Training file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    last_count_report = 0.0

    def report_count_progress(completed_chunks: int, total_chunks: int, processed_bytes: int, unique_pretokens: int) -> None:
        nonlocal last_count_report
        now = time.time()
        if completed_chunks == total_chunks or now - last_count_report >= 5:
            last_count_report = now
            print(
                "[count] "
                f"{completed_chunks}/{total_chunks} chunks, "
                f"{_format_bytes(processed_bytes)}, "
                f"{unique_pretokens:,} unique pre-tokens",
                flush=True,
            )

    def report_merge_progress(learned_merges: int, total_merges: int, best_pair: Pair, best_pair_count: int) -> None:
        left, right = best_pair
        print(
            "[merge] "
            f"{learned_merges:,}/{total_merges:,}, "
            f"count={best_pair_count:,}, "
            f"pair=({left.hex()}, {right.hex()})",
            flush=True,
        )

    print(f"Training BPE from {input_path}", flush=True)
    print(
        f"vocab_size={args.vocab_size:,}, workers={args.num_workers}, "
        f"chunk_size={args.chunk_size_mb} MiB, special_tokens={args.special_tokens}",
        flush=True,
    )
    _print_resource_hint(input_path, args.num_workers, args.chunk_size_mb)

    vocab, merges = train_bpe_from_file(
        input_path,
        args.vocab_size,
        args.special_tokens,
        num_workers=args.num_workers,
        chunk_size_mb=args.chunk_size_mb,
        split_special_token=split_special_token,
        max_bytes=args.max_bytes,
        count_progress_callback=report_count_progress,
        merge_progress_callback=report_merge_progress,
        merge_progress_every=args.merge_progress_every,
    )

    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    metadata_path = output_dir / "metadata.json"
    _write_vocab(vocab, vocab_path)
    _write_merges(merges, merges_path)
    metadata_path.write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "vocab_size": args.vocab_size,
                "special_tokens": args.special_tokens,
                "num_merges": len(merges),
                "num_workers": args.num_workers,
                "chunk_size_mb": args.chunk_size_mb,
                "split_special_token": split_special_token,
                "max_bytes": args.max_bytes,
                "elapsed_seconds": round(time.time() - started_at, 3),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {vocab_path}", flush=True)
    print(f"Wrote {merges_path}", flush=True)
    print(f"Wrote {metadata_path}", flush=True)
    print(f"Done in {time.time() - started_at:.1f}s", flush=True)


if __name__ == "__main__":
    main()
