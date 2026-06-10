from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import wandb, json, os

# Load id2label
with open("id2label.json") as f:
    id2label = json.load(f)
label2id = {v: k for k, v in id2label.items()}

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=6,
    id2label=id2label, label2id=label2id
)

# Load dataset
dataset = load_dataset("dair-ai/emotion", "split")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True,
                     padding="max_length", max_length=128)

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")}

wandb.init(project="mlops-assignment3", name="run-v1",
    config={"model": model_name, "epochs": 3, "batch_size": 16,
            "learning_rate": 3e-5, "version": "v1"})

args = TrainingArguments(
    output_dir="./results", num_train_epochs=3,
    per_device_train_batch_size=16, eval_strategy="epoch",
    save_strategy="epoch", load_best_model_at_end=True,
    report_to="wandb", run_name="run-v1", learning_rate=3e-5,
)

trainer = Trainer(model=model, args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics)

trainer.train()
wandb.finish()
print("Training complete!")
