import os
import re
import random
import argparse
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)


def clean_text(text):
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords(text, max_keywords=6):
    text = str(text).lower()
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)

    words = [w for w in words if w not in ENGLISH_STOP_WORDS]

    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, count in sorted_words[:max_keywords]]

    if not keywords:
        keywords = ["intelligent", "research", "system"]

    return keywords


def generate_new_topic(title, domain, summary):
    domain = clean_text(domain)
    title = clean_text(title)
    summary = clean_text(summary)

    keywords = extract_keywords(summary, max_keywords=5)
    main_phrase = " ".join(keywords[:4]).title()

    patterns = [
        f"An Intelligent Framework for {main_phrase} in {domain}",
        f"A Deep Learning Approach for {main_phrase} in {domain}",
        f"Context-Aware {domain} System for {main_phrase}",
        f"An Explainable AI Framework for {main_phrase} in {domain}",
        f"Advanced {domain} Research Using {main_phrase}",
        f"A Transformer-Based Approach for {main_phrase} in {domain}",
        f"Adaptive Machine Learning Framework for {main_phrase} in {domain}",
        f"A Data-Driven Approach for {main_phrase} in {domain}",
        f"Intelligent Prediction Framework for {main_phrase} in {domain}",
        f"Deep Learning-Based Analysis of {main_phrase} in {domain}"
    ]

    new_topic = random.choice(patterns)

    if new_topic.lower().strip() == title.lower().strip():
        new_topic = f"Advanced {domain} Framework for {main_phrase}"

    return new_topic


def build_prompt(title, domain, summary):
    prompt = f"""
Generate a completely new academic research topic.

Rules:
- Do not copy the original title.
- Use the summary as the main source.
- Use the domain as the second most important source.
- Use the original title only as a weak reference.
- Output only one professional research topic.

Domain:
{domain}

Summary:
{summary}

Original Title:
{title}

New Research Topic:
"""
    return clean_text(prompt)


def prepare_dataset(csv_path, title_col, domain_col, summary_col, target_col=None, subset_size=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print("\nAvailable columns:")
    print(list(df.columns))

    required_cols = [title_col, domain_col, summary_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available columns: {list(df.columns)}"
            )

    df = df.dropna(subset=required_cols)

    if subset_size:
        df = df.sample(min(subset_size, len(df)), random_state=42)

    df[title_col] = df[title_col].apply(clean_text)
    df[domain_col] = df[domain_col].apply(clean_text)
    df[summary_col] = df[summary_col].apply(clean_text)

    df["input_text"] = df.apply(
        lambda row: build_prompt(
            row[title_col],
            row[domain_col],
            row[summary_col]
        ),
        axis=1
    )

    if target_col and target_col in df.columns:
        print(f"\nUsing target column: {target_col}")
        df["target_text"] = df[target_col].apply(clean_text)
    else:
        print("\nNo target column found. Creating synthetic new research topics...")
        df["target_text"] = df.apply(
            lambda row: generate_new_topic(
                row[title_col],
                row[domain_col],
                row[summary_col]
            ),
            axis=1
        )

    df = df[
        df["target_text"].str.lower().str.strip()
        != df[title_col].str.lower().str.strip()
    ]

    return df[["input_text", "target_text"]]


def tokenize_function(batch, tokenizer, max_input_length=512, max_target_length=64):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=max_input_length,
        truncation=True
    )

    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=max_target_length,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def train_model(args):
    print("\nLoading dataset...")

    df = prepare_dataset(
        csv_path=args.data_path,
        title_col=args.title_col,
        domain_col=args.domain_col,
        summary_col=args.summary_col,
        target_col=args.target_col,
        subset_size=args.subset_size
    )

    print("\nDataset prepared successfully.")
    print(df.head())

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    print("\nLoading model and tokenizer...")
    print("Model:", args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nUsing device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model.to(device)

    tokenized_train = train_dataset.map(
        lambda batch: tokenize_function(
            batch,
            tokenizer,
            args.max_input_length,
            args.max_target_length
        ),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_val = val_dataset.map(
        lambda batch: tokenize_function(
            batch,
            tokenizer,
            args.max_input_length,
            args.max_target_length
        ),
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
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
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

    print("\nSaving final model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nTraining completed successfully.")
    print("Model saved to:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--title_col", type=str, default="title")
    parser.add_argument("--domain_col", type=str, default="category")
    parser.add_argument("--summary_col", type=str, default="summary")
    parser.add_argument("--target_col", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--output_dir", type=str, default="./flant5_topic_model")

    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)

    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=64)

    args = parser.parse_args()

    train_model(args)