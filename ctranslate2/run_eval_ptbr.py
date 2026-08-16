"""Run evaluation for ctranslate2 whisper models in PT-BR (batched inference)."""
import argparse
import dataclasses
import os
import time

import evaluate
import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline
from tqdm import tqdm

from normalizer import data_utils

wer_metric = evaluate.load("wer")


def main(args) -> None:
    """Main function to run evaluation on a dataset."""
    model = WhisperModel(
        model_size_or_path=args.model_id,
        compute_type="float16",
        device="cuda",
        device_index=args.device,
    )
    pipeline = BatchedInferencePipeline(model=model)

    def transcribe_audio(audio_array: np.ndarray) -> str:
        segments, _ = pipeline.transcribe(
            audio_array,
            language="pt",
            batch_size=args.batch_size,
        )
        return "".join(dataclasses.asdict(s)["text"] for s in segments).strip()

    # ── Warmup ──────────────────────────────────────────────────────────────
    if args.warmup_steps is not None and args.warmup_steps > 0:
        print(f"Warming up with {args.warmup_steps} samples...")
        warmup_dataset = data_utils.load_data_ptbr(args)
        warmup_dataset = data_utils.prepare_data_ptbr(warmup_dataset)
        warmup_samples: list = []
        for sample in warmup_dataset:
            warmup_samples.append(sample)
            if len(warmup_samples) >= args.warmup_steps:
                break
        for sample in warmup_samples:
            transcribe_audio(np.array(sample["audio"]["array"]))
        print("Warmup done.")

    # ── Load & prepare dataset ───────────────────────────────────────────────
    dataset = data_utils.load_data_ptbr(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data_ptbr(dataset)

    # ── Collect all samples then transcribe ──────────────────────────────────
    samples = list(tqdm(dataset, desc="Loading samples"))

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }

    for sample in tqdm(samples, desc="Transcribing"):
        audio_array = np.array(sample["audio"]["array"])
        sr = sample["audio"]["sampling_rate"]
        audio_length = len(audio_array) / sr

        start = time.time()
        prediction = transcribe_audio(audio_array)
        elapsed = time.time() - start

        all_results["audio_length_s"].append(audio_length)
        all_results["transcription_time_s"].append(elapsed)
        all_results["predictions"].append(prediction)
        all_results["references"].append(sample["original_text"])

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest_model_name = args.model_name if args.model_name else args.model_id
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        manifest_model_name,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    norm_refs  = [data_utils.ml_normalizer(r, lang="pt") for r in all_results["references"]]
    norm_preds = [data_utils.ml_normalizer(p, lang="pt") for p in all_results["predictions"]]
    wer = wer_metric.compute(references=norm_refs, predictions=norm_preds)
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a faster-whisper model on the PT-BR ASR leaderboard datasets."
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with faster-whisper.",
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
        help="Dataset path (HuggingFace repo ID).",
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
        help="Dataset split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device index.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for internal chunked batching inside BatchedInferencePipeline.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit evaluation to this many samples.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset lazily.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of samples to use for warm-up before timed runs.",
    )
    args = parser.parse_args()

    print("*" * 100)
    print(f"Evaluating {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split} [PT-BR]")
    print("*" * 100)

    main(args)
