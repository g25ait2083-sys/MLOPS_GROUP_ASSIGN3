from transformers import pipeline
import os

# Load model from Hugging Face Hub
model_name = os.environ.get(
    "HF_MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english"  # default until you push yours
)

input_text = os.environ.get("INPUT_TEXT", "This movie was absolutely amazing!")

print(f"Loading model: {model_name}")
print(f"Input text: {input_text}")

# Run inference
classifier = pipeline(
    "text-classification",
    model=model_name,
    token=os.environ.get("HF_TOKEN")
)

result = classifier(input_text)

print(f"Prediction: {result[0]['label']}")
print(f"Confidence: {round(result[0]['score'] * 100, 2)}%")
