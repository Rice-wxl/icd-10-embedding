"""Download trained models from the HF repo into Model/ for local use.

Reverses the upload-time rename so that downstream scripts find files at
the paths config.py already expects:

    HF mortality_30day/mort_*.keras       → Model/mort_nodie_*.keras
    HF readmission_30day/readmit_*.keras  → Model/readmit_*.keras
    HF encoders/*.pkl                     → Model/*.pkl

After running this, scripts that resolve `MODEL_PATH` via `src/config.py`
(evaluate, calibration, DeLong, IG, inference) work without code changes.

Auth: HF_TOKEN in env, or `huggingface-cli login`, only needed for private
repos.

Usage:
    python scripts/download_from_hf.py --repo-id <user-or-org>/<repo-name>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODEL_DIR = Path("/users/xwang259/icd/Model")

# (HF subfolder, HF filename prefix, local filename prefix).
# Mirrors PREFIX_RULES in upload_to_hf.py — keep these in sync.
DOWNLOAD_RULES = [
    ("mortality_30day",   "mort_",    "mort_nodie_"),
    ("readmission_30day", "readmit_", "readmit_"),
]

ENCODER_FILES = ["full_label_encoder.pkl", "full_age_scaler.pkl"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-id", required=True,
                    help="HF repo, e.g. xwang259/nrd-icd-outcomes")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help=f"Local Model/ directory (default: {DEFAULT_MODEL_DIR})")
    ap.add_argument("--revision", default=None,
                    help="HF branch/tag/commit to pull (default: main)")
    args = ap.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching snapshot of {args.repo_id}…")
    snapshot = Path(snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        allow_patterns=["*.keras", "*.pkl"],
    ))

    n = 0
    for hf_sub, hf_prefix, local_prefix in DOWNLOAD_RULES:
        src_dir = snapshot / hf_sub
        if not src_dir.is_dir():
            print(f"  WARNING missing on HF: {hf_sub}/", file=sys.stderr)
            continue
        for src in sorted(src_dir.glob("*.keras")):
            if not src.name.startswith(hf_prefix):
                print(f"  skip (unexpected prefix): {hf_sub}/{src.name}",
                      file=sys.stderr)
                continue
            local_name = local_prefix + src.name[len(hf_prefix):]
            dst = args.model_dir / local_name
            shutil.copy2(src, dst)
            print(f"  {hf_sub}/{src.name} → Model/{local_name}")
            n += 1

    enc_src_dir = snapshot / "encoders"
    if enc_src_dir.is_dir():
        for enc in ENCODER_FILES:
            src = enc_src_dir / enc
            if not src.exists():
                print(f"  WARNING missing on HF: encoders/{enc}",
                      file=sys.stderr)
                continue
            shutil.copy2(src, args.model_dir / enc)
            print(f"  encoders/{enc} → Model/{enc}")
            n += 1
    else:
        print("  WARNING missing on HF: encoders/", file=sys.stderr)

    print(f"Downloaded {n} files into {args.model_dir}/")


if __name__ == "__main__":
    main()
