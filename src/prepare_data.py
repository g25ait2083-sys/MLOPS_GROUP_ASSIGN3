from datasets import load_dataset
import json, re

dataset = load_dataset("dair-ai/emotion", "split")

# Inspect
print("Train size:", len(dataset["train"]))
print("Labels:", dataset["train"].features["label"].names)

# Clean
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")

def clean_text(example):
    t = example["text"].lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    example["text"] = " ".join(t.split()).strip()
    return example

dataset = dataset.map(clean_text)
dataset = dataset.filter(lambda x: len(x["text"]) > 0)

# Save id2label
class_label = dataset["train"].features["label"]
id2label = {str(i): n for i, n in enumerate(class_label.names)}
with open("id2label.json", "w") as f:
    json.dump(id2label, f, indent=2)

print("Done! Labels:", id2label)
