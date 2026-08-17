#!/usr/bin/env python3
"""Download all results from an HF bucket, score them, and save a results.csv.

Usage:
    python scripts/score_bucket_results.py
    python scripts/score_bucket_results.py --bucket <bucket>
    python scripts/score_bucket_results.py --bucket <bucket> --local_dir results
    python scripts/score_bucket_results.py --skip_download   # re-score already-downloaded results
    python scripts/score_bucket_results.py --skip_upload   # skip uploading results.csv to the bucket after scoring.
"""

import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile

# Allow importing normalizer from the repo root regardless of where the script
# is called from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from normalizer.eval_utils import score_results


def sync_bucket(bucket: str, local_dir: str, hf_token: str | None = None) -> None:
    """Sync an HF bucket to a local directory using the `hf` CLI."""
    bucket_url = f"hf://buckets/{bucket}"
    print(f"Syncing {bucket_url}  →  {local_dir} ...")
    os.makedirs(local_dir, exist_ok=True)
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    subprocess.run(
        ["hf", "buckets", "sync", bucket_url, local_dir],
        check=True,
        env=env,
    )
    print("Sync complete.\n")


def upload_csv_to_bucket(
    csv_path: str, bucket: str, hf_token: str | None = None
) -> None:
    """Upload results.csv to the root of the HF bucket."""
    bucket_url = f"hf://buckets/{bucket}"
    print(f"Uploading {csv_path}  →  {bucket_url}/results.csv ...")
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    # hf buckets sync only works with directories, so stage the file in a tmpdir
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy2(csv_path, os.path.join(tmpdir, "results.csv"))
        subprocess.run(
            ["hf", "buckets", "sync", tmpdir, bucket_url],
            check=True,
            env=env,
        )
    print("Upload complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Score all results from an HF bucket and save results.csv."
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="HF bucket name (without the hf://buckets/ prefix).",
    )
    parser.add_argument(
        "--local_dir",
        default=None,
        help="Local directory to sync results into. Defaults to <repo_root>/results.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip syncing the bucket and score already-downloaded results in --local_dir.",
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Skip uploading results.csv to the bucket after scoring.",
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token for private buckets. Defaults to $HF_TOKEN env var.",
    )
    parser.add_argument(
        "--model_id",
        action="append",
        default=None,
        metavar="MODEL_ID",
        help="Score only this model (can be repeated for multiple models). "
        "E.g. --model_id zoom/scribe_v1 --model_id assembly/universal-3-pro. "
        "Defaults to scoring all models.",
    )
    args = parser.parse_args()

    bucket = args.bucket
    local_dir = args.local_dir or os.path.join(REPO_ROOT, "results")

    if not args.skip_download:
        sync_bucket(bucket, local_dir, hf_token=args.hf_token)
    else:
        print(f"Skipping download — scoring results in: {local_dir}\n")

    if not os.path.isdir(local_dir):
        print(f"ERROR: Local results directory not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    model_ids = args.model_id or [None]  # None means all models

    # Capture CSV output so we can both print it and save it to a file
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for model_id in model_ids:
            try:
                score_results(
                    local_dir,
                    model_id=model_id,
                    csv_only=True,
                )
            except ValueError as e:
                print(f"Skipping model_id={model_id}: {e}")

    csv_content = buf.getvalue()
    sys.stdout.write(csv_content)  # still print to the terminal (with decorations)

    # Keep only valid CSV lines (header + data rows all contain a comma;
    # decorative *** borders and title lines do not)
    csv_lines = [ln for ln in csv_content.splitlines() if "," in ln]
    clean_csv = "\n".join(csv_lines) + "\n"

    # Save to results.csv inside local_dir
    csv_path = os.path.join(local_dir, "results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(clean_csv)
    print(f"\nCSV saved to: {csv_path}")

    # Upload results.csv to the bucket
    if not args.skip_upload:
        upload_csv_to_bucket(csv_path, bucket, hf_token=args.hf_token)


if __name__ == "__main__":
    main()
