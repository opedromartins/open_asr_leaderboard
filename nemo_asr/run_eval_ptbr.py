import argparse

import io
import os
import subprocess
import torch
import evaluate
import soundfile

from tqdm import tqdm
from normalizer import data_utils
import numpy as np

from nemo.collections.asr.models import ASRModel
import time

# HF bucket where PT-BR results are stored
PTBR_BUCKET = "opedromartins/asr-leaderboard-5080"

wer_metric = evaluate.load("wer")


def main(args):

    data_cache_root = (
        args.data_cache_root if args.data_cache_root is not None else os.getcwd()
    )
    DATA_CACHE_DIR = os.path.join(data_cache_root, "audio_cache")
    DATASET_NAME = args.dataset
    SPLIT_NAME = args.split

    CACHE_DIR = os.path.join(DATA_CACHE_DIR, DATASET_NAME, SPLIT_NAME)
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
        compute_dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        compute_dtype = torch.float32

    if args.model_id.endswith(".nemo"):
        asr_model = ASRModel.restore_from(args.model_id, map_location=device)
    else:
        asr_model = ASRModel.from_pretrained(
            args.model_id, map_location=device
        )  # type: ASRModel

    asr_model.to(compute_dtype)
    asr_model.eval()
    print(
        f"Model size: {sum(p.numel() for p in asr_model.parameters()) / 1e9:.2f}B parameters"
    )

    # Load dataset using PT-BR loader (same as load_data but documents intent)
    dataset = data_utils.load_data_ptbr(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        dataset = dataset.take(args.max_eval_samples)

    # Prepare data with PT-BR normalization (uses clean_transcription as reference)
    dataset = data_utils.prepare_data_ptbr(dataset)

    def download_audio_files(batch):
        audio_paths = []
        original_audio_paths = []
        durations = []

        # Use audio_filename (PT-BR dataset column) or fall back to id
        if "audio_filename" in batch:
            file_names = batch["audio_filename"]
        else:
            file_names = batch.get("file_name", [None] * len(batch["audio"]))

        # Generate sequential IDs if no id column
        if "id" in batch:
            ids = batch["id"]
        else:
            start_idx = (
                len([f for f in os.listdir(CACHE_DIR) if f.endswith(".wav")])
                if os.path.exists(CACHE_DIR)
                else 0
            )
            ids = [f"sample_{start_idx + i}" for i in range(len(batch["audio"]))]

        for id, file_name, audio_sample in zip(ids, file_names, batch["audio"]):
            original_id = id
            id = id.replace("/", "_").removesuffix(".wav")

            audio_path = os.path.join(CACHE_DIR, f"{id}.wav")
            audio_array = np.float32(audio_sample["array"])
            sample_rate = audio_sample["sampling_rate"]

            if not os.path.exists(audio_path):
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                soundfile.write(audio_path, audio_array, sample_rate)

            audio_paths.append(audio_path)
            if file_name is not None:
                original_audio_paths.append(os.path.basename(str(file_name)))
            else:
                original_audio_paths.append(original_id)
            durations.append(len(audio_array) / sample_rate)

        batch["references"] = batch[
            "original_text"
        ]  # clean_transcription; normalization applied at scoring time
        batch["audio_filepaths"] = audio_paths
        batch["original_audio_filepaths"] = original_audio_paths
        batch["durations"] = durations

        return batch

    if asr_model.cfg.decoding.strategy != "beam":
        asr_model.cfg.decoding.strategy = "greedy_batch"
        asr_model.change_decoding_strategy(asr_model.cfg.decoding)

    dataset = dataset.map(
        download_audio_files,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
    )

    all_data = {
        "audio_filepaths": [],
        "original_audio_filepaths": [],
        "durations": [],
        "references": [],
    }

    data_itr = iter(dataset)
    for data in tqdm(data_itr, desc="Downloading Samples"):
        for key in all_data:
            all_data[key].append(data[key])

    # Sort by duration (longest first) for efficient batching
    sorted_indices = sorted(
        range(len(all_data["durations"])),
        key=lambda k: all_data["durations"][k],
        reverse=True,
    )
    all_data["audio_filepaths"] = [
        all_data["audio_filepaths"][i] for i in sorted_indices
    ]
    all_data["original_audio_filepaths"] = [
        all_data["original_audio_filepaths"][i] for i in sorted_indices
    ]
    all_data["references"] = [all_data["references"][i] for i in sorted_indices]
    all_data["durations"] = [all_data["durations"][i] for i in sorted_indices]

    total_time = 0
    for _ in range(2):  # warmup once and calculate RTFx
        if _ == 0:
            audio_files = all_data["audio_filepaths"][: args.batch_size * 4]
        else:
            audio_files = all_data["audio_filepaths"]
        start_time = time.time()
        with torch.inference_mode(), torch.no_grad():
            if "canary" in args.model_id and "v2" not in args.model_id:
                pnc = "nopnc"
            else:
                pnc = "pnc"

            if "canary" in args.model_id:
                transcriptions = asr_model.transcribe(
                    audio_files,
                    batch_size=args.batch_size,
                    verbose=False,
                    pnc=pnc,
                    num_workers=1,
                )
            else:
                transcriptions = asr_model.transcribe(
                    audio_files,
                    batch_size=args.batch_size,
                    verbose=False,
                    num_workers=1,
                )
        end_time = time.time()
        if _ == 1:
            total_time += end_time - start_time

    if isinstance(transcriptions, tuple) and len(transcriptions) == 2:
        transcriptions = transcriptions[0]
    predictions = [pred.text for pred in transcriptions]

    avg_time = total_time / len(all_data["audio_filepaths"])

    manifest_model_name = args.model_name if args.model_name else args.model_id
    manifest_path = data_utils.write_manifest(
        all_data["references"],
        predictions,
        manifest_model_name,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_data["durations"],
        transcription_time=[avg_time] * len(all_data["audio_filepaths"]),
        audio_filepaths=all_data["original_audio_filepaths"],
    )

    print("Results saved at path:", os.path.abspath(manifest_path))

    norm_references = [
        data_utils.ml_normalizer(r, lang="pt") for r in all_data["references"]
    ]
    norm_predictions = [data_utils.ml_normalizer(p, lang="pt") for p in predictions]
    wer = wer_metric.compute(references=norm_references, predictions=norm_predictions)
    wer = round(100 * wer, 2)

    audio_length = sum(all_data["durations"])
    rtfx = audio_length / total_time
    rtfx = round(rtfx, 2)

    print("RTFX:", rtfx)
    print("WER:", wer, "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a NeMo ASR model on the PT-BR leaderboard datasets."
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with NVIDIA NeMo.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for output files. Defaults to model_id.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="opedromartins/asr-leaderboard-datasets-ptbr",
        help="Dataset path (HuggingFace repo ID). Defaults to opedromartins/asr-leaderboard-datasets-ptbr",
    )
    parser.add_argument(
        "--data_cache_root",
        type=str,
        default=None,
        help="Root directory for audio cache. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset subset name. One of: coraa-mupe, coraa-nurc-sp, coraa-v1.1, fleurs, mls, tedx",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split. Defaults to 'test'.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device to run on. -1 for CPU, 0 for first GPU, etc.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit evaluation to this many samples (useful for testing).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset lazily instead of downloading it in full.",
    )
    args = parser.parse_args()

    print("*" * 100)
    print(
        f"Evaluating {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split} [PT-BR]"
    )
    print("*" * 100)

    main(args)
