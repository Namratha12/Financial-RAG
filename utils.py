# utils.py

import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def format_prompt(prompt: str) -> str:
    """Clean and format prompt before sending to LLM."""
    return prompt.strip().replace("\n\n", "\n")


def calc_metrics(y_true, y_pred, positive_label="supported"):
    """Calculate accuracy, precision, recall, F1."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
