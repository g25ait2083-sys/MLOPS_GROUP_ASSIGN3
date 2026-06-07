from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import wandb
import json
import os

# Load id2label
with open("id2label.json") as f:
    id2label = json.load(f)
label2id = {v: k for k, v in id2label.items()}

# Model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# Load & tokenize dataset
dataset = load_dataset("stanfordnlp/imdb")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True,
                     padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Use a small subset for Kaggle free GPU
train_dataset = tokenized["train"].select(range(5000))
eval_dataset  = tokenized["test"].select(range(1000))

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }

# W&B init
wandb.init(
    project="mlops-assignment3",
    name="run-v1",
    config={
        "model":         model_name,
        "epochs":        2,
        "batch_size":    16,
        "learning_rate": 3e-5,
        "version":       "v1",
        "platform":      "Kaggle",
    }
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="run-v1",
    learning_rate=3e-5,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
wandb.finish()
print("Training complete!")
