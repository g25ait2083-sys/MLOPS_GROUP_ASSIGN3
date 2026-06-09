"""Task 2 — Data preparation & normalisation for the emotion classifier.

This is a reusable, command-line script. It downloads the public
``dair-ai/emotion`` dataset, reports its structure and class balance, applies a
light text-normalisation pass appropriate for short social-media style English,
builds the ``id2label`` mapping straight from the dataset's own label schema, and
writes the cleaned splits to disk as Parquet.

Only ``data/id2label.json`` is meant to be committed to git — the prepared
Parquet files are large and are covered by ``.gitignore``.

Usage
-----
    python src/data_prep.py --output-dir data/prepared

Design notes (see report for the full justification):
  * We lowercase and strip URLs / @mentions because emotion in this corpus is
    carried by words, not links or handles, and DistilBERT-*uncased* lowercases
    internally anyway — doing it here keeps the saved data faithful to what the
    model sees.
  * We collapse repeated whitespace and drop empty / duplicate rows so identical
    samples cannot leak across the train/validation boundary.
  * We deliberately keep sentence punctuation by default (``--strip-punctuation``
    is opt-in): exclamation marks and question marks are genuine emotion signal.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# datasets / pandas are dev-time deps (see requirements-dev.txt); they are not
# needed at inference time so we import them lazily inside main().

DEFAULT_DATASET = "dair-ai/emotion"
DEFAULT_CONFIG = "split"  # 16k train / 2k validation / 2k test
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP_PATH = REPO_ROOT / "data" / "id2label.json"

# Pre-compiled patterns — compiling once is cheaper when cleaning ~20k rows.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s']")  # keep word chars, whitespace and apostrophes


def clean_text(text: str, strip_punctuation: bool = False) -> str:
    """Normalise a single example. Returns an empty string for unusable input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    if strip_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_label_mapping(class_label) -> dict[int, str]:
    """Derive {id: name} from the dataset's own ClassLabel feature.

    Building the map from the dataset (rather than hardcoding it) guarantees the
    integer ids saved here always line up with the ids used during training.
    """
    return {idx: name for idx, name in enumerate(class_label.names)}


def report_distribution(name: str, labels, id2label: dict[int, str]) -> None:
    counts = Counter(labels)
    total = sum(counts.values())
    print(f"\n[{name}] {total} samples")
    for idx in sorted(counts):
        n = counts[idx]
        print(f"  {idx2name(id2label, idx):<10} {n:>6}  ({100 * n / total:5.1f}%)")


def idx2name(id2label: dict[int, str], idx: int) -> str:
    # id2label keys may be ints (in-memory) or strings (after JSON round-trip).
    return id2label.get(idx, id2label.get(str(idx), str(idx)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the emotion dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default="data/prepared")
    parser.add_argument(
        "--strip-punctuation",
        action="store_true",
        help="Also remove punctuation (off by default — '!' / '?' carry emotion).",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.dataset} ({args.config}) ...")
    raw = load_dataset(args.dataset, args.config)
    print(raw)

    class_label = raw["train"].features[LABEL_COLUMN]
    id2label = build_label_mapping(class_label)
    print(f"\nDiscovered {len(id2label)} labels: {id2label}")

    # --- Inspect raw class distribution before touching anything. ---
    for split in raw:
        report_distribution(f"raw:{split}", raw[split][LABEL_COLUMN], id2label)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in raw:
        df = raw[split].to_pandas()[[TEXT_COLUMN, LABEL_COLUMN]]
        before = len(df)

        # 1) Handle missing values.
        df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        n_missing = before - len(df)

        # 2) Normalise text.
        df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(
            lambda t: clean_text(t, strip_punctuation=args.strip_punctuation)
        )

        # 3) Drop rows that became empty after cleaning.
        df = df[df[TEXT_COLUMN].str.len() > 0]

        # 4) Drop exact duplicate texts (keep the first occurrence).
        before_dedup = len(df)
        df = df.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)
        n_dupes = before_dedup - len(df)

        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

        out_path = output_dir / f"{split}.parquet"
        df.to_parquet(out_path, index=False)

        summary[split] = {
            "rows_in": before,
            "rows_out": len(df),
            "dropped_missing": n_missing,
            "dropped_duplicates": n_dupes,
        }
        print(
            f"\n[clean:{split}] {before} -> {len(df)} rows "
            f"(missing={n_missing}, duplicates={n_dupes}) -> {out_path}"
        )

    # --- Persist the label mapping (the one artefact we commit to git). ---
    LABEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABEL_MAP_PATH.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)
        f.write("\n")
    print(f"\nSaved label mapping -> {LABEL_MAP_PATH}")

    print("\nCleaning summary:")
    print(json.dumps(summary, indent=2))
    print("\nDone. Prepared splits are git-ignored; commit only data/id2label.json.")


if __name__ == "__main__":
    main()
