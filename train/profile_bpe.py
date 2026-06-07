"""
Profile BPE training: memory & time per phase.
Uses owt_valid.txt (277MB) with vocab_size=32000.
Includes periodic memory snapshots during the merge loop.
"""
import time
import tracemalloc
import sys
from pathlib import Path
from collections import Counter, defaultdict
import heapq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.train_bpe import (
    count_pretoken_frequencies,
    BYTE_TOKENS,
    Reverse_Pair,
)


def train_bpe_profiled(token_counts, vocab_size, special_tokens=None, snapshot_interval=500):
    """Copy of train_bpe_from_token_counts with periodic memory snapshots."""
    vocab = {i: BYTE_TOKENS[i] for i in range(256)}
    next_id = 256
    for token in (special_tokens or []):
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1
    if vocab_size <= len(vocab):
        return {idx: vocab[idx] for idx in range(vocab_size)}, []

    pair_counts = Counter()
    pair_to_tokens = defaultdict(set)

    for token, count in token_counts.items():
        for pair, occurrences in Counter(zip(token, token[1:])).items():
            pair_counts[pair] += count * occurrences
            pair_to_tokens[pair].add(token)

    pair_heap = [(-count, Reverse_Pair(pair), pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    print(f"  [init]   vocab={len(vocab)}  heap={len(pair_heap):,}  "
          f"pair_counts={len(pair_counts):,}  pair_to_tokens_keys={len(pair_to_tokens):,}  "
          f"current_mem={current_mem/1024/1024:.1f}MB  peak={peak_mem/1024/1024:.1f}MB")

    snapshot_time = time.time()
    merges = []
    merge_count = 0

    while len(vocab) < vocab_size and pair_heap:
        best_pair = None
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
        merge_count += 1

        new_token_counts = Counter()
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
                    heapq.heappush(pair_heap, (-pair_counts[pair], Reverse_Pair(pair), pair))

            result = []
            i = 0
            while i < len(token):
                if i+1 < len(token) and token[i:i+2] == best_pair:
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
                heapq.heappush(pair_heap, (-pair_counts[pair], Reverse_Pair(pair), pair))

        if merge_count % snapshot_interval == 0:
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            elapsed = time.time() - snapshot_time
            print(f"  [merge {merge_count:>5}/{vocab_size-256}]  heap={len(pair_heap):>10,}  "
                  f"token_counts={len(token_counts):>8,}  pair_counts={len(pair_counts):>8,}  "
                  f"current_mem={current_mem/1024/1024:>8.1f}MB  peak={peak_mem/1024/1024:>8.1f}MB  "
                  f"elapsed={elapsed:.1f}s")

    return vocab, merges


def main():
    file_path = "tests/fixtures/tinystories_sample_5M.txt"  # 5MB for quick profiling
    vocab_size = 8000  # smaller for quick profiling
    special_tokens = [""]
    snapshot_interval = 500

    file_size_mb = Path(file_path).stat().st_size / 1024 / 1024
    print(f"Dataset: {file_path} ({file_size_mb:.1f} MB)")
    print(f"Target vocab size: {vocab_size}")

    # ── Step 1: Pretokenization ──
    print(f"\n{'='*70}")
    print(f"  Step 1: Pretokenization")
    print(f"{'='*70}")
    tracemalloc.start()
    t0 = time.time()

    token_counts = count_pretoken_frequencies(file_path, special_tokens, num_workers=1)

    t1 = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Time: {t1 - t0:.2f}s")
    print(f"  Unique pretokens: {len(token_counts):,}")
    print(f"  Peak memory (tracemalloc): {peak / 1024 / 1024:.1f} MB")

    # ── Step 2: BPE training (profiled) ──
    print(f"\n{'='*70}")
    print(f"  Step 2: BPE Training (vocab_size={vocab_size})")
    print(f"{'='*70}")
    tracemalloc.start()
    t2 = time.time()

    vocab, merges = train_bpe_profiled(token_counts, vocab_size, special_tokens, snapshot_interval)

    t3 = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n  Final: vocab={len(vocab)}, merges={len(merges)}")
    print(f"  Time: {t3 - t2:.2f}s")
    print(f"  Peak memory (tracemalloc): {peak / 1024 / 1024:.1f} MB")
    print(f"\n  Total time: {t3 - t0:.2f}s")

    # ── Extrapolation ──
    owt_size_gb = 11.0
    ratio = owt_size_gb / (file_size_mb / 1024)
    print(f"\n{'='*70}")
    print(f"  Extrapolation to owt_train.txt ({owt_size_gb}GB)")
    print(f"{'='*70}")
    print(f"  Size ratio: {ratio:.0f}x")
    print(f"  Estimated pretoken time: {(t1-t0)*ratio/60:.0f} min (linear scaling)")
    print(f"  Estimated BPE time: {(t3-t2)*ratio/60:.0f} ~ {(t3-t2)*ratio**1.5/60:.0f} min (linear~superlinear)")
    print(f"  Note: memory will also scale roughly proportionally!")


if __name__ == "__main__":
    main()
