"""
Research Topic Classifier - SciBERT with Domain Embedding
Follows the Colab workflow (Scibert_Workflow1.ipynb) exactly.

Train on Colab (GPU recommended), then bring the zip here.

Local quick test (CPU):
    python train_topic_generator.py --nrows 5000 --min_topic_count 10 --epochs 1 --batch_size 4

Full training (Colab GPU):
    python train_topic_generator.py --nrows 50000 --min_topic_count 50 --epochs 5 --batch_size 8
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments


# ── Data loading (exactly as in PDF) ─────────────────────────────────────────

def load_data(nrows: int = 50000) -> pd.DataFrame:
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    file_name = os.listdir(path)[0]
    file_path = os.path.join(path, file_name)
    df = pd.read_json(file_path, lines=True, nrows=nrows)
    return df


# ── Preprocessing (exactly as in PDF) ────────────────────────────────────────

def preprocess(df: pd.DataFrame, min_topic_count: int = 50):
    df = df[df["categories"].apply(lambda x: len(x.split()) > 1)]
    df["domain"] = df["categories"].apply(lambda x: x.split()[0])
    df["topic"]  = df["categories"].apply(lambda x: x.split()[1])

    topic_counts  = df["topic"].value_counts()
    valid_topics  = topic_counts[topic_counts >= min_topic_count].index
    df = df[df["topic"].isin(valid_topics)]
    print(f"Dataset size after filtering rare topics: {len(df)}")

    df["text"] = df["title"] + " " + df["abstract"]

    topic_encoder          = LabelEncoder()
    df["label"]            = topic_encoder.fit_transform(df["topic"])
    num_labels             = len(topic_encoder.classes_)
    print("Topic Classes:", num_labels)

    domain_encoder         = LabelEncoder()
    df["domain_encoded"]   = domain_encoder.fit_transform(df["domain"])
    num_domains            = len(domain_encoder.classes_)
    print("Domain Classes:", num_domains)

    return df, topic_encoder, domain_encoder, num_labels, num_domains


# ── Dataset class (exactly as in PDF) ────────────────────────────────────────

class ArxivDataset(Dataset):
    def __init__(self, encodings, labels, domains):
        self.encodings = encodings
        self.labels    = labels
        self.domains   = domains

    def __getitem__(self, idx):
        item             = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"]   = torch.tensor(self.labels[idx])
        item["domain"]   = torch.tensor(self.domains[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ── Model (exactly as in PDF) ─────────────────────────────────────────────────

class SciBERTWithDomain(nn.Module):
    def __init__(self, num_labels, num_domains):
        super().__init__()
        self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

        # domain embedding size increased to 128
        self.domain_embedding = nn.Embedding(num_domains, 128)

        # hidden layer with ReLU and Dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask, domain, labels=None, **kwargs):
        outputs     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output  = outputs.last_hidden_state[:, 0, :]
        domain_embed = self.domain_embedding(domain)
        combined    = torch.cat((cls_output, domain_embed), dim=1)
        logits      = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss    = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# ── Metrics (exactly as in PDF) ───────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds          = logits.argmax(axis=1)
    acc            = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows",           type=int,   default=50000)
    parser.add_argument("--min_topic_count", type=int,   default=50)
    parser.add_argument("--epochs",          type=int,   default=5)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--learning_rate",   type=float, default=3e-5)
    parser.add_argument("--output_dir",      type=str,   default="scibert_topic_model")
    args = parser.parse_args()

    # 1. Load
    df = load_data(nrows=args.nrows)

    # 2. Preprocess
    df, topic_encoder, domain_encoder, num_labels, num_domains = preprocess(
        df, min_topic_count=args.min_topic_count
    )

    # 3. Split
    (train_texts, test_texts,
     train_labels, test_labels,
     train_domains, test_domains) = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        df["domain_encoded"].tolist(),
        test_size=0.2,
        random_state=42,
    )

    # 4. Tokenize
    tokenizer       = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=256)

    train_dataset = ArxivDataset(train_encodings, train_labels, train_domains)
    test_dataset  = ArxivDataset(test_encodings,  test_labels,  test_domains)

    # 5. Model
    model = SciBERTWithDomain(num_labels, num_domains)
    for param in model.parameters():
        param.data = param.data.contiguous()

    # 6. Training arguments (from PDF)
    training_args = TrainingArguments(
        output_dir                  = "./results",
        learning_rate               = args.learning_rate,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        num_train_epochs            = args.epochs,
        weight_decay                = 0.01,
        logging_dir                 = "./logs",
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        report_to                   = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = test_dataset,
        compute_metrics = compute_metrics,
    )

    # 7. Train
    trainer.train()

    # 8. Evaluate
    results = trainer.evaluate()
    print(results)

    # 9. Confusion matrix (from PDF)
    predictions = trainer.predict(test_dataset)
    preds       = predictions.predictions.argmax(axis=1)

    cm = confusion_matrix(test_labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.show()

    # 10. Classification report (from PDF)
    print("--- Detailed Classification Report ---")
    report = classification_report(test_labels, preds, target_names=topic_encoder.classes_)
    print(report)

    # 11. Save model + encoders so the app can load them
    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), save_dir / "model_state.pt")
    tokenizer.save_pretrained(save_dir)

    with open(save_dir / "topic_encoder.pkl", "wb") as f:
        pickle.dump(topic_encoder, f)
    with open(save_dir / "domain_encoder.pkl", "wb") as f:
        pickle.dump(domain_encoder, f)
    with open(save_dir / "app_config.json", "w") as f:
        json.dump({"num_labels": num_labels, "num_domains": num_domains}, f)

    print(f"\nModel saved to ./{args.output_dir}/")
    print("Zip this folder and extract it locally to use with app.py")


if __name__ == "__main__":
    main()
