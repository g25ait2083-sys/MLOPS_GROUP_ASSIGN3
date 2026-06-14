"""Source package for the emotion-classification MLOps pipeline.

Modules
-------
data_prep    : download, inspect, clean and persist the emotion dataset.
model_utils  : load the tokenizer and DistilBERT model with the right labels.
inference    : load the fine-tuned model from the Hub and classify one text.
"""

__all__ = ["data_prep", "model_utils", "inference"]
__version__ = "1.0.0"
