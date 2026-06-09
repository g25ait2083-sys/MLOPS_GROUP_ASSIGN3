"""Tasks 6/7 — Stand-alone inference entry point.

Loads the fine-tuned emotion model from the Hugging Face Hub and classifies a
single piece of text. Configuration is entirely through environment variables so
the exact same script runs locally, inside the Docker container, and inside the
GitHub Actions ``inference.yml`` workflow.

Environment variables
----------------------
HF_MODEL_NAME : Hub repo id of the fine-tuned model (overridable; has a default).
INPUT_TEXT    : the text to classify (required).
HF_TOKEN      : optional — only needed if the model repo is private.

The script intentionally depends only on packages in ``requirements.txt`` (torch,
transformers, huggingface_hub, numpy) and reads the label names from the model's
own config, so the container needs no dataset files.
"""

from __future__ import annotations

import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# A default so the image is runnable out of the box. It points at a public,
# label-compatible emotion model (same 6 classes). Override it with
# `-e HF_MODEL_NAME=<your-username>/distilbert-emotion-mlops-a3` once you have
# pushed your own fine-tuned model in Task 5.
DEFAULT_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"


def load_pipeline(model_name: str, token: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
    model.eval()
    return tokenizer, model


def classify(text: str, tokenizer, model) -> dict:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    # Prefer the human-readable names baked into the model config.
    id2label = model.config.id2label
    top_id = int(torch.argmax(probs).item())

    distribution = {
        id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))
    }
    return {
        "input": text,
        "predicted_label": id2label[top_id],
        "confidence": round(float(probs[top_id]), 4),
        "distribution": dict(
            sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }


def main() -> int:
    model_name = os.environ.get("HF_MODEL_NAME", DEFAULT_MODEL)
    text = os.environ.get("INPUT_TEXT")
    token = os.environ.get("HF_TOKEN")  # None for public repos — that's fine.

    if not text:
        print("ERROR: set INPUT_TEXT (the text to classify).", file=sys.stderr)
        return 2

    print(f"Loading model: {model_name}", file=sys.stderr)
    tokenizer, model = load_pipeline(model_name, token)
    result = classify(text, tokenizer, model)

    # Machine-readable result to stdout so CI / logs can capture it cleanly.
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
