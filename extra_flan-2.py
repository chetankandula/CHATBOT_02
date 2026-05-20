import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def clean_text(text):
    text = str(text).strip().lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9,;:()\-./ ]+", " ", text)
    return text.strip()


def shorten_summary(text, max_words=160):
    return " ".join(text.split()[:max_words])


def build_prompt(category, summary):
    category = clean_text(category)
    summary = shorten_summary(clean_text(summary), 160)

    return (
        "generate a concise meaningful academic research topic. "
        f"category: {category}. "
        f"summary: {summary}"
    )


def prepare_dataset(csv_path, subset_size=None):
    df = pd.read_csv(csv_path)

    required_cols = ["title", "summary", "category"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[required_cols].dropna().drop_duplicates()

    df["title_clean"] = df["title"].apply(clean_text)
    df["summary_clean"] = df["summary"].apply(clean_text)
    df["category_clean"] = df["category"].apply(clean_text)

    df = df[df["title_clean"].str.split().str.len() >= 4]
    df = df[df["summary_clean"].str.split().str.len() >= 30]

    df["input_text"] = df.apply(
        lambda row: build_prompt(
            row["category_clean"],
            row["summary_clean"]
        ),
        axis=1
    )

    # Output = real research title from dataset
    df["target_text"] = df["title_clean"]

    if subset_size:
        df = df.sample(min(subset_size, len(df)), random_state=42)

    return df[["input_text", "target_text"]].reset_index(drop=True)


def tokenize_function(batch, tokenizer, max_input_length=384, max_target_length=64):
    inputs = tokenizer(
        batch["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = [
        [token if token != tokenizer.pad_token_id else -100 for token in label]
        for label in labels["input_ids"]
    ]

    return inputs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="arXiv_scientific dataset.csv")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--output_dir", type=str, default="flant5_category_summary_topic_model")
    parser.add_argument("--subset_size", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nPreparing dataset...")
    df = prepare_dataset(args.data_path, args.subset_size)

    print("Dataset size:", df.shape)
    print(df.head())

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)

    tokenized_train = train_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_val = val_dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_steps=50,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator
    )

    print("\nTraining started...")
    trainer.train()

    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nTraining completed.")
    print("Model saved to:", args.output_dir)


if __name__ == "__main__":
    main()