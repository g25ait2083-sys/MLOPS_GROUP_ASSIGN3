from datasets import load_dataset
import json

# Load IMDb dataset
dataset = load_dataset("stanfordnlp/imdb")

# Inspect
print("Train size:", len(dataset["train"]))
print("Test size:", len(dataset["test"]))
print("Sample:", dataset["train"][0])

# Clean training data
def clean_example(example):
    example["text"] = example["text"].lower().strip()
    return example

dataset = dataset.map(clean_example)

# Remove duplicates
train_texts = set()
def is_unique(example):
    if example["text"] in train_texts:
        return False
    train_texts.add(example["text"])
    return True

dataset["train"] = dataset["train"].filter(is_unique)

# Save id2label mapping
id2label = {"0": "negative", "1": "positive"}
with open("id2label.json", "w") as f:
    json.dump(id2label, f)

print("Cleaning done!")
print("Final train size:", len(dataset["train"]))
