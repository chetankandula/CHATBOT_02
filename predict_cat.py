import argparse
import re
from pathlib import Path

import joblib
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


def clean_text(text):
    text = str(text).strip().lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9,;:()\-./ ]+", " ", text)
    return text.strip()


def main():

    parser = argparse.ArgumentParser(
        description="SciBERT Category Prediction"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="scibert_category_model"
    )

    parser.add_argument(
        "--title",
        type=str,
        required=True
    )

    parser.add_argument(
        "--summary",
        type=str,
        required=True
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if not model_dir.exists():

        raise FileNotFoundError(
            f"Model folder not found: {model_dir}\n"
            "Train the model first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading model...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir
    )

    label_encoder = joblib.load(
        model_dir / "label_encoder.joblib"
    )

    model.to(device)

    model.eval()

    text = (
        clean_text(args.title)
        + " "
        + clean_text(args.summary)
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_index
        ].item()

    predicted_category = (
        label_encoder.inverse_transform(
            [predicted_index]
        )[0]
    )

    print("\nPredicted Category:")
    print(predicted_category)

    print("\nConfidence:")
    print(f"{confidence*100:.2f}%")



if __name__ == "__main__":
    main()