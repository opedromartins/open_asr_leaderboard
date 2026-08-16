#!/bin/bash
# Evaluate Parakeet TDT 0.6B v3 on the PT-BR ASR leaderboard datasets.
#
# Usage:
#   bash run_parakeet_ptbr.sh                          # full evaluation
#   MAX_EVAL_SAMPLES=64 bash run_parakeet_ptbr.sh      # quick smoke-test
#
# Requires: uv (https://docs.astral.sh/uv/)
# Dependencies are declared in pyproject.toml under [dependency-groups]
# and installed with: uv sync --group nemo-ptbr

export PYTHONPATH="..":$PYTHONPATH
# Ensure uv is on PATH (common install location)
export PATH="$HOME/.local/bin:$PATH"

BATCH_SIZE=128
DEVICE_ID=0
BUCKET_REPO="opedromartins/asr-leaderboard-5080"

# ── Install / sync the nemo dependency group ───────────────────────────
echo "Syncing nemo dependencies..."
uv sync --group nemo
echo "Sync complete."

# ─────────────────── Models ───────────────────
MODEL_IDs=(
    "nvidia/parakeet-tdt-0.6b-v3"
)

# ── Datasets: "subset split" (all subsets of opedromartins/asr-leaderboard-datasets-ptbr)
DATASET_CONFIGS=(
    "coraa-mupe test"
    "coraa-nurc-sp test"
    "coraa-v1.1 test"
    "fleurs test"
    "mls test"
    "tedx test"
)

# Optional: limit number of samples for smoke-testing (set via env var)
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"   # -1 means all samples

for MODEL_ID in "${MODEL_IDs[@]}"; do

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"

        echo "=========================================="
        echo "Model: ${MODEL_ID}"
        echo "Dataset: ${DATASET} / ${SPLIT}"
        echo "=========================================="

        uv run python run_eval_ptbr.py \
            --model_id="${MODEL_ID}" \
            --dataset_path="opedromartins/asr-leaderboard-datasets-ptbr" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --device=${DEVICE_ID} \
            --batch_size=${BATCH_SIZE} \
            --max_eval_samples=${MAX_EVAL_SAMPLES}
    done

    # Move generated results into a model-specific folder
    MODEL_FOLDER="${MODEL_ID//\//-}"
    mkdir -p "./results/${MODEL_FOLDER}"
    mv ./results/*.jsonl "./results/${MODEL_FOLDER}/" 2>/dev/null || true


    # Score results locally after all subsets are done
    uv run python -c "
import sys; sys.path.insert(0, '..')
from normalizer.eval_utils import score_results
score_results(
    '$(pwd)/results',
    model_id='${MODEL_ID}'
)"

done

# ── Upload results at the end ───────────────────────────────────────────────
echo "=========================================="
echo "Uploading results to bucket..."
echo "=========================================="
uv run hf buckets sync ./results "hf://buckets/${BUCKET_REPO}"
echo "Upload complete."
