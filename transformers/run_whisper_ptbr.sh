#!/bin/bash
# Evaluate Whisper models (🤗 Transformers) on the PT-BR ASR leaderboard datasets.
#
# Usage:
#   bash run_whisper_ptbr.sh                          # full evaluation
#   MAX_EVAL_SAMPLES=64 bash run_whisper_ptbr.sh      # quick smoke-test
#
# Requires: uv (https://docs.astral.sh/uv/)
# Dependencies are declared in pyproject.toml under [dependency-groups]
# and installed with: uv sync --group transformers

export PYTHONPATH="..":$PYTHONPATH
# Ensure uv is on PATH (common install location)
export PATH="$HOME/.local/bin:$PATH"

DEVICE_ID=0
BATCH_SIZE=32
BUCKET_REPO="opedromartins/asr-leaderboard-5080"
DATASET_PATH="opedromartins/asr-leaderboard-datasets-ptbr"

# ── Install / sync the transformers dependency group ──────────────────────────
echo "Syncing transformers dependencies..."
uv sync --group transformers
echo "Sync complete."

# ────────────────────────────────────────────────────────────────────────────
# Models to evaluate
# ────────────────────────────────────────────────────────────────────────────
MODEL_IDs=(
    "openai/whisper-tiny"
    "openai/whisper-small"
    "openai/whisper-base"
    "openai/whisper-medium"
    "openai/whisper-large"
    "openai/whisper-large-v2"
    "openai/whisper-large-v3"
    "openai/whisper-large-v3-turbo"
    "distil-whisper/distil-large-v3"
    "distil-whisper/distil-large-v3.5"
)

# ── Datasets: "subset split" ──────────────────────────────────────────────────
DATASET_CONFIGS=(
    "coraa-mupe test"
    "coraa-nurc-sp test"
    "coraa-v1.1 test"
    "fleurs test"
    "mls test"
    "tedx test"
)

MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"   # -1 means all samples

for MODEL_ID in "${MODEL_IDs[@]}"; do

    MODEL_FOLDER="${MODEL_ID//\//-}"
    MODEL_SLUG="${MODEL_ID//\//-}"
    DATASET_PATH_SLUG="${DATASET_PATH//\//-}"
    mkdir -p "./results/${MODEL_FOLDER}"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"

        # Compute the filename that write_manifest will generate
        RESULT_FILE="./results/${MODEL_FOLDER}/MODEL_${MODEL_SLUG}_DATASET_${DATASET_PATH_SLUG}_${DATASET}_${SPLIT}.jsonl"

        if [ -f "${RESULT_FILE}" ]; then
            echo "Skipping ${MODEL_ID} / ${DATASET} / ${SPLIT} — result already exists."
            continue
        fi

        echo "=========================================="
        echo "Model: ${MODEL_ID}"
        echo "Dataset: ${DATASET} / ${SPLIT}"
        echo "=========================================="

        uv run python run_eval_ptbr.py \
            --model_id="${MODEL_ID}" \
            --dataset_path="${DATASET_PATH}" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --device=${DEVICE_ID} \
            --batch_size=${BATCH_SIZE} \
            --max_eval_samples=${MAX_EVAL_SAMPLES}

        # Move the generated file directly into the model folder
        mv "./results/MODEL_${MODEL_SLUG}_DATASET_${DATASET_PATH_SLUG}_${DATASET}_${SPLIT}.jsonl" \
           "./results/${MODEL_FOLDER}/" 2>/dev/null || true
    done

    # Local score summary
    uv run python -c "
import sys; sys.path.insert(0, '..')
from normalizer.eval_utils import score_results
score_results(
    '$(pwd)/results',
    model_id='${MODEL_ID}'
)"

    # Sync this model's results folder immediately after evaluation
    echo "Syncing results for ${MODEL_ID}..."
    uv run hf buckets sync "./results/${MODEL_FOLDER}" "hf://buckets/${BUCKET_REPO}/${MODEL_FOLDER}"
    echo "Sync complete for ${MODEL_ID}."

done

# ── Upload results at the end ─────────────────────────────────────────────────
echo "=========================================="
echo "Uploading results to bucket..."
echo "=========================================="
uv run hf buckets sync ./results "hf://buckets/${BUCKET_REPO}"
echo "Upload complete."
