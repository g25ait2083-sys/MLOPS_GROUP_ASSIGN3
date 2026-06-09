"""Task 3 — Load the tokenizer and DistilBERT model from the Hugging Face Hub.

Kept deliberately small and dependency-light so the same helpers can be reused by
the training notebook (Task 4) and, where relevant, the inference path (Task 6).

The number of output labels and the id<->label maps are read from
``data/id2label.json`` so the classification head always matches the data
prepared in Task 2 — change the dataset there and this stays correct with no edits.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_MAP = REPO_ROOT / "data" / "id2label.json"
DEFAULT_BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 128  # emotion samples are short tweets; 128 tokens is plenty.


def load_label_maps(path: Path | str = DEFAULT_LABEL_MAP):
    """Return (id2label, label2id) with integer keys for id2label."""
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)
    id2label = {int(k): v for k, v in raw.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def load_tokenizer(base_model: str = DEFAULT_BASE_MODEL):
    """Load the tokenizer for the base model (Task 3, step 1)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_model)


def load_model(base_model: str = DEFAULT_BASE_MODEL, label_map_path=DEFAULT_LABEL_MAP):
    """Load the sequence-classification model with the correct number of labels.

    Passing ``id2label`` / ``label2id`` into the config means the saved model (and
    therefore the inference container) reports human-readable class names instead
    of ``LABEL_0`` … ``LABEL_5`` (Task 3, step 2).
    """
    from transformers import AutoModelForSequenceClassification

    id2label, label2id = load_label_maps(label_map_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    return model


def make_tokenize_fn(tokenizer, text_column: str = "text", max_length: int = MAX_LENGTH):
    """Return a function suitable for ``datasets.Dataset.map(..., batched=True)``."""

    def _tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return _tokenize


if __name__ == "__main__":
    # Quick smoke test: confirms the head size matches the label map.
    tok = load_tokenizer()
    mdl = load_model()
    id2label, _ = load_label_maps()
    print(f"Tokenizer: {tok.__class__.__name__}")
    print(f"Model:     {mdl.__class__.__name__}")
    print(f"Labels ({mdl.config.num_labels}): {id2label}")
    assert mdl.config.num_labels == len(id2label), "Head size != label-map size!"
    print("OK — model head matches id2label.json")
