import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def clean_text(text):
    text = str(text).strip().lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9,;:()\-./ ]+", " ", text)
    return text.strip()


def prepare_dataset(csv_path, subset_size=None, min_category_count=20):
    df = pd.read_csv(csv_path)

    required_cols = ["title", "summary", "category"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required_cols].dropna().drop_duplicates().reset_index(drop=True)

    df["title_clean"] = df["title"].apply(clean_text)
    df["summary_clean"] = df["summary"].apply(clean_text)
    df["category_clean"] = df["category"].apply(clean_text)

    df = df[df["title_clean"].str.split().str.len() >= 3]
    df = df[df["summary_clean"].str.split().str.len() >= 20]

    df["text"] = df["title_clean"] + " " + df["summary_clean"]

    if subset_size:
        df = df.sample(min(subset_size, len(df)), random_state=42)

    category_counts = df["category_clean"].value_counts()

    valid_categories = category_counts[
        category_counts >= min_category_count
    ].index

    df = df[df["category_clean"].isin(valid_categories)].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No categories left after filtering. "
            "Lower --min_category_count."
        )

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["category_clean"])

    return df[["text", "label"]], label_encoder


def tokenize_function(batch, tokenizer, max_length=256):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="arXiv_scientific dataset.csv")
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--output_dir", type=str, default="scibert_category_model")
    parser.add_argument("--subset_size", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--min_category_count", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nPreparing dataset...")

    df, label_encoder = prepare_dataset(
        csv_path=args.data_path,
        subset_size=args.subset_size,
        min_category_count=args.min_category_count,
    )

    print("Dataset size after filtering:", df.shape)
    print("Number of categories:", len(label_encoder.classes_))

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    train_dataset = train_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
    )

    test_dataset = test_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
    )

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_encoder.classes_),
    )

    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=50,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("\nTraining started...")
    trainer.train()

    print("\nEvaluating model...")
    predictions = trainer.predict(test_dataset)

    y_pred = predictions.predictions.argmax(axis=1)
    y_true = predictions.label_ids

    print("\nAccuracy:", accuracy_score(y_true, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    joblib.dump(label_encoder, f"{args.output_dir}/label_encoder.joblib")

    print("\nTraining completed successfully.")
    print("Model saved to:", args.output_dir)


if __name__ == "__main__":
    main()