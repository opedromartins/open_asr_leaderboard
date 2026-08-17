"""Run evaluation for Transformers Whisper models on the PT-BR ASR leaderboard datasets."""

import argparse
import os
import random
import re

import evaluate
import numpy as np
import torch
from normalizer import data_utils
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from transformers import (
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_MULTIMODAL_LM_MAPPING,
    MODEL_FOR_RNNT_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    AutoConfig,
    AutoModelForCTC,
    AutoModelForMultimodalLM,
    AutoModelForRNNT,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    CompileConfig,
)

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("high")


def main(args) -> None:
    """Main function to run evaluation on a PT-BR dataset."""
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    torch_dtype = getattr(torch, args.dtype)

    config = AutoConfig.from_pretrained(args.model_id, revision=args.revision)
    if type(config) in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING:
        cls_model = AutoModelForSpeechSeq2Seq
    elif type(config) in MODEL_FOR_MULTIMODAL_LM_MAPPING:
        cls_model = AutoModelForMultimodalLM
    elif type(config) in MODEL_FOR_CTC_MAPPING:
        cls_model = AutoModelForCTC
    elif type(config) in MODEL_FOR_RNNT_MAPPING:
        cls_model = AutoModelForRNNT
    else:
        raise ValueError(
            f"Model config of type {type(config)} not recognized in Transformers mappings."
        )
    is_ctc = cls_model == AutoModelForCTC

    model = cls_model.from_pretrained(
        args.model_id,
        dtype=torch_dtype,
        revision=args.revision,
        attn_implementation=args.attn_implementation,
    )
    model.to(args.device)
    model.eval()
    print(
        f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters"
    )

    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)

    # Extract sampling rate
    if (
        hasattr(processor, "feature_extractor")
        and processor.feature_extractor is not None
    ):
        sampling_rate = processor.feature_extractor.sampling_rate
    elif (
        hasattr(processor, "audio_processor") and processor.audio_processor is not None
    ):
        sampling_rate = processor.audio_processor.sampling_rate
    else:
        sampling_rate = 16_000

    # Set generate arguments
    if model.can_generate():
        gen_kwargs = {}
        if args.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = args.max_new_tokens
        if getattr(model.generation_config, "is_multilingual", False):
            gen_kwargs["language"] = "pt"
            gen_kwargs["task"] = "transcribe"
        # Clear deprecated Whisper generation config fields to suppress warnings
        if hasattr(model.generation_config, "forced_decoder_ids"):
            model.generation_config.forced_decoder_ids = None
        if hasattr(model.generation_config, "suppress_tokens"):
            model.generation_config.suppress_tokens = []
        if hasattr(model.generation_config, "begin_suppress_tokens"):
            model.generation_config.begin_suppress_tokens = []
    elif args.max_new_tokens:
        raise ValueError(
            "`max_new_tokens` should only be set for auto-regressive models, but got a CTC model."
        )

    if args.torch_compile is not None:
        if model.can_generate():
            gen_kwargs["compile_config"] = CompileConfig(
                mode=args.torch_compile, fullgraph=args.compile_fullgraph
            )
            model.generation_config.cache_implementation = "static"
        else:
            model = torch.compile(
                model, mode=args.torch_compile, fullgraph=args.compile_fullgraph
            )

        # Ensure warm-up runs when using torch.compile
        if args.warmup_steps is None or args.warmup_steps < 1:
            print(
                "`--torch_compile` is enabled; forcing `--warmup_steps=10` to trigger compilation before timed runs."
            )
            args.warmup_steps = 10

    def benchmark(batch, min_new_tokens=None):
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)
        sr = batch["audio"][0]["sampling_rate"]
        batch["audio_length_s"] = [len(audio) / sr for audio in audios]
        batch["audio_filepath"] = data_utils.extract_audio_filepaths_from_batch(
            batch, minibatch_size
        )

        # START TIMING
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # 1. Pre-Processing
        padding_size = None
        if minibatch_size != args.batch_size and args.torch_compile is not None:
            padding_size = args.batch_size - minibatch_size
            padding_audios = [audios[-1] for _ in range(padding_size)]
            audios.extend(padding_audios)

        if not model.can_generate():
            # CTC: normalize to mean 0, std 1
            inputs = processor(
                audios,
                sampling_rate=sr,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            )
        else:
            # Standard Whisper: pad to 30s, convert to log-mel
            inputs = processor(
                audios,
                sampling_rate=sr,
                return_tensors="pt",
                padding="longest",
                return_attention_mask=True,
                device=args.device,
            )

        inputs = inputs.to(args.device, dtype=torch_dtype)

        # 2. Model Inference
        if args.torch_compile is not None:
            sdpa_backends = [SDPBackend.MATH]
        else:
            sdpa_backends = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        with sdpa_kernel(sdpa_backends):
            if model.can_generate():
                pred_ids = model.generate(
                    **inputs, **gen_kwargs, min_new_tokens=min_new_tokens
                )
            else:
                with torch.no_grad():
                    logits = model(**inputs).logits
                    pred_ids = logits.argmax(-1)

        # 3. Post-processing
        if padding_size is not None:
            pred_ids = pred_ids[:-padding_size, ...]

        if is_ctc:
            # Don't use skip_special_tokens as it collapses double letters
            pred_text = processor.batch_decode(pred_ids)
        else:
            pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0

        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = pred_text  # raw; normalization applied at scoring time
        batch["references"] = batch["original_text"]  # PT-BR reference column
        return batch

    # ── Warmup ──────────────────────────────────────────────────────────────
    if args.warmup_steps is not None and args.warmup_steps > 0:
        warmup_dataset = data_utils.load_data_ptbr(args)
        warmup_dataset = data_utils.prepare_data_ptbr(
            warmup_dataset, sampling_rate=sampling_rate
        )

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(
                range(min(num_warmup_samples, len(warmup_dataset)))
            )
        warmup_dataset = iter(
            warmup_dataset.map(
                benchmark,
                batch_size=args.batch_size,
                batched=True,
                fn_kwargs={"min_new_tokens": args.max_new_tokens},
            )
        )
        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    # ── Load & prepare dataset ───────────────────────────────────────────────
    dataset = data_utils.load_data_ptbr(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    dataset = data_utils.prepare_data_ptbr(dataset, sampling_rate=sampling_rate)

    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
        "audio_filepath": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

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
        audio_filepaths=all_results["audio_filepath"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    norm_refs = [
        data_utils.ml_normalizer(r, lang="pt") for r in all_results["references"]
    ]
    norm_preds = [
        data_utils.ml_normalizer(p, lang="pt") for p in all_results["predictions"]
    ]
    wer = wer_metric.compute(references=norm_refs, predictions=norm_preds)
    wer = round(100 * wer, 2)
    rtfx = round(
        sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2
    )
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Transformers Whisper model on the PT-BR ASR leaderboard datasets."
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers.",
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
        help="GPU device index. -1 for CPU.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
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
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (for auto-regressive models).",
    )
    parser.add_argument(
        "--torch_compile",
        type=str,
        default=None,
        help="Mode for torch.compile. E.g. 'default', 'reduce-overhead', 'max-autotune'.",
    )
    parser.add_argument(
        "--compile_fullgraph",
        action="store_true",
        help="Whether to do full graph compilation.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="dtype for model loading and inference. E.g. 'float16', 'bfloat16', 'float32'.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="Attention implementation. E.g. 'sdpa', 'eager', 'flash_attention_2'.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warm-up steps before timed runs.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision to use. Defaults to the main branch.",
    )
    args = parser.parse_args()

    print("*" * 100)
    print(
        f"Evaluating {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split} [PT-BR]"
    )
    print("*" * 100)

    main(args)
