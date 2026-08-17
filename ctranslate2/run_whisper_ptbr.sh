#!/bin/bash
# Evaluate Whisper models (faster-whisper / ctranslate2) on the PT-BR ASR leaderboard datasets.
#
# Supports two kinds of model IDs:
#   1. Already converted to CT2 format (available directly on the Hub):
#      listed in CT2_MODEL_IDs — used as-is with faster-whisper.
#   2. Standard HF Transformers models that need conversion first:
#      listed in HF_MODEL_IDs — converted locally with ct2-transformers-converter,
#      then evaluated from the local output directory.
#
# Usage:
#   bash run_whisper_ptbr.sh                          # full evaluation
#   MAX_EVAL_SAMPLES=64 bash run_whisper_ptbr.sh      # quick smoke-test
#
# Requires: uv (https://docs.astral.sh/uv/)
# Dependencies are declared in pyproject.toml under [dependency-groups]
# and installed with: uv sync --group ctranslate2

export PYTHONPATH="..":$PYTHONPATH
# Ensure uv is on PATH (common install location)
export PATH="$HOME/.local/bin:$PATH"

DEVICE_ID=0
BATCH_SIZE=16
BUCKET_REPO="opedromartins/asr-leaderboard-5080"
DATASET_PATH="opedromartins/asr-leaderboard-datasets-ptbr"

# Directory where locally converted CT2 models are stored
CT2_CONVERT_DIR="./ct2_models"

# ── Install / sync the ctranslate2 dependency group ───────────────────────────
echo "Syncing ctranslate2 dependencies..."
uv sync --group ctranslate2
echo "Sync complete."

# ────────────────────────────────────────────────────────────────────────────
# List 1: already in CT2 format on the Hub
#         These are passed directly to faster-whisper as model_id.
# ────────────────────────────────────────────────────────────────────────────
CT2_MODEL_IDs=(
    # "Systran/faster-whisper-large-v3"
)

# ────────────────────────────────────────────────────────────────────────────
# List 2: standard HF Transformers models — converted on first run
#         Each entry is just the HuggingFace model ID (e.g. "openai/whisper-large-v3").
#         The output directory is generated automatically.
# ────────────────────────────────────────────────────────────────────────────
HF_MODEL_IDs=(
    "openai/whisper-tiny"
    "openai/whisper-small"
    "openai/whisper-base"
    "openai/whisper-medium"
    "openai/whisper-large"
    "openai/whisper-large-v2"
    "openai/whisper-large-v3"
    "openai/whisper-large-v3-turbo"
)

# ── Datasets ─────────────────────────────────────────────────────────────────
DATASET_CONFIGS=(
    "coraa-mupe test"
    "coraa-nurc-sp test"
    "coraa-v1.1 test"
    "fleurs test"
    "mls test"
    "tedx test"
)

MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"   # -1 means all samples

# ── Helper: evaluate a single model_id ───────────────────────────────────────
run_eval() {
    local MODEL_ID="$1"      # used for naming results folder and scoring
    local MODEL_PATH="$2"    # passed to faster-whisper (Hub ID or local path)

    local MODEL_FOLDER="${MODEL_ID//\//-}"
    local MODEL_SLUG="${MODEL_ID//\//-}"
    local DATASET_PATH_SLUG="${DATASET_PATH//\//-}"
    mkdir -p "./results/${MODEL_FOLDER}"

    for cfg in "${DATASET_CONFIGS[@]}"; do
        read -r DATASET SPLIT <<< "$cfg"

        # Compute the filename that write_manifest will generate
        local RESULT_FILE="./results/${MODEL_FOLDER}/MODEL_${MODEL_SLUG}_DATASET_${DATASET_PATH_SLUG}_${DATASET}_${SPLIT}.jsonl"

        if [ -f "${RESULT_FILE}" ]; then
            echo "Skipping ${MODEL_ID} / ${DATASET} / ${SPLIT} — result already exists."
            continue
        fi

        echo "=========================================="
        echo "Model: ${MODEL_ID}"
        echo "Dataset: ${DATASET} / ${SPLIT}"
        echo "=========================================="

        uv run python run_eval_ptbr.py \
            --model_id="${MODEL_PATH}" \
            --model_name="${MODEL_ID}" \
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
}

# ── Evaluate models already in CT2 format ────────────────────────────────────
for MODEL_ID in "${CT2_MODEL_IDs[@]}"; do
    run_eval "$MODEL_ID" "$MODEL_ID"
done

# ── Convert + evaluate HF Transformers models ─────────────────────────────────
mkdir -p "${CT2_CONVERT_DIR}"

for HF_ID in "${HF_MODEL_IDs[@]}"; do
    OUTPUT_SUBDIR="${HF_ID//\//-}-ct2"
    LOCAL_PATH="${CT2_CONVERT_DIR}/${OUTPUT_SUBDIR}"

    if [ ! -d "${LOCAL_PATH}" ]; then
        echo "=========================================="
        echo "Converting ${HF_ID} -> ${LOCAL_PATH}"
        echo "=========================================="
        
        HAS_TOKENIZER=$(uv run python -c "from huggingface_hub import file_exists; print(file_exists('${HF_ID}', 'tokenizer.json'))" 2>/dev/null || echo "False")
        
        COPY_FILES="preprocessor_config.json"
        if [ "$HAS_TOKENIZER" = "True" ]; then
            COPY_FILES="preprocessor_config.json tokenizer.json"
        fi

        uv run ct2-transformers-converter \
            --model "${HF_ID}" \
            --output_dir "${LOCAL_PATH}" \
            --copy_files $COPY_FILES \
            --quantization float16
    else
        echo "Skipping conversion — ${LOCAL_PATH} already exists."
    fi

    run_eval "$HF_ID" "$LOCAL_PATH"
done

# ── Upload results at the end ─────────────────────────────────────────────────
echo "=========================================="
echo "Uploading results to bucket..."
echo "=========================================="
uv run hf buckets sync ./results "hf://buckets/${BUCKET_REPO}"
echo "Upload complete."
