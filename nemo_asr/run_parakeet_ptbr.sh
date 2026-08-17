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
DATASET_PATH="opedromartins/asr-leaderboard-datasets-ptbr"

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

    MODEL_FOLDER="${MODEL_ID//\//_}"
    mkdir -p "./results/${MODEL_FOLDER}"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"

        # Compute the filename that write_manifest will generate
        DATASET_PATH_SLUG="${DATASET_PATH//\//-}"
        MODEL_SLUG="${MODEL_ID//\//_}"
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
            --model_name="${MODEL_SLUG}" \
            --dataset_path="${DATASET_PATH}" \
            --dataset="${DATASET}" \
            --split="${SPLIT}" \
            --device=${DEVICE_ID} \
            --batch_size=${BATCH_SIZE} \
            --max_eval_samples=${MAX_EVAL_SAMPLES}

        # Move the generated file directly into the model folder
        mv ./results/MODEL_${MODEL_SLUG}_DATASET_${DATASET_PATH_SLUG}_${DATASET}_${SPLIT}.jsonl \
           "./results/${MODEL_FOLDER}/" 2>/dev/null || true
    done


    # Score results locally after all subsets are done
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

# ── Upload results at the end ───────────────────────────────────────────────
echo "=========================================="
echo "Uploading results to bucket..."
echo "=========================================="
uv run hf buckets sync ./results "hf://buckets/${BUCKET_REPO}"
echo "Upload complete."
