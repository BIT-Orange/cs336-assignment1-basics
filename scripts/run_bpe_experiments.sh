#!/usr/bin/env bash
set -euo pipefail

cs336_data_dir="${CS336_DATA_DIR:-/content/drive/MyDrive/CS336_Data}"
cs336_output_root="${CS336_OUTPUT_ROOT:-${cs336_data_dir}/BPE_Results}"
cs336_workers="${CS336_WORKERS:-8}"
cs336_run_tag="${CS336_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
cs336_run_dir="${cs336_output_root}/${cs336_run_tag}"

mkdir -p "${cs336_run_dir}"
git rev-parse HEAD > "${cs336_run_dir}/git_commit.txt"

python -m pip install --quiet 'psutil>=7' 'regex>=2026.3.32'

python -m cs336_basics.bpe_experiment \
    --input-path "${cs336_data_dir}/TinyStoriesV2-GPT4-train.txt" \
    --output-dir "${cs336_run_dir}/tinystories" \
    --vocab-size 10000 \
    --special-token '<|endoftext|>' \
    --num-workers "${cs336_workers}" \
    --token-counts-cache "${cs336_run_dir}/tinystories/token_counts.pkl" \
    2>&1 | tee "${cs336_run_dir}/tinystories.log"

python -m cs336_basics.bpe_experiment \
    --input-path "${cs336_data_dir}/TinyStoriesV2-GPT4-train.txt" \
    --output-dir "${cs336_run_dir}/tinystories-profile" \
    --vocab-size 10000 \
    --special-token '<|endoftext|>' \
    --num-workers "${cs336_workers}" \
    --token-counts-cache "${cs336_run_dir}/tinystories/token_counts.pkl" \
    --profile \
    2>&1 | tee "${cs336_run_dir}/tinystories-profile.log"

python -m cs336_basics.bpe_experiment \
    --input-path "${cs336_data_dir}/owt_train.txt" \
    --output-dir "${cs336_run_dir}/openwebtext" \
    --vocab-size 32000 \
    --special-token '<|endoftext|>' \
    --num-workers "${cs336_workers}" \
    --token-counts-cache "${cs336_run_dir}/openwebtext/token_counts.pkl" \
    2>&1 | tee "${cs336_run_dir}/openwebtext.log"

echo "BPE results saved to ${cs336_run_dir}"
