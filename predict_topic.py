"""
Predict research topic using the trained SciBERT model.

Run:
    python predict_topic.py --title "..." --domain "cs.AI" --abstract "..."
"""

import argparse
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer


# ── Model (matches actual saved weights) ─────────────────────────────────────

class SciBERTWithDomain(nn.Module):
    def __init__(self, num_labels, num_domains, bert_config=None):
        super().__init__()
        self.bert             = AutoModel.from_config(bert_config) if bert_config \
                                else AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.domain_embedding = nn.Embedding(num_domains, 128)
        self.text_bottleneck  = nn.Sequential(nn.Linear(768, 256))
        self.classifier       = nn.Sequential(
            nn.Linear(384, 128),   # 256 (text) + 128 (domain) = 384
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_labels),
        )

    def forward(self, input_ids, attention_mask, domain, labels=None, **kwargs):
        cls_output   = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_out     = self.text_bottleneck(cls_output)
        domain_embed = self.domain_embedding(domain)
        logits       = self.classifier(torch.cat((text_out, domain_embed), dim=1))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
        return {"loss": loss, "logits": logits}


# ── Loader ────────────────────────────────────────────────────────────────────

def load_model(model_dir: str = "scibert_topic_model"):
    import numpy as np

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model folder not found: {model_dir.resolve()}\n"
            "Train on Colab, zip scibert_topic_model/ and extract it here."
        )

    with open(model_dir / "app_config.json") as f:
        config = json.load(f)

    topic_classes  = np.load(model_dir / "classes.npy", allow_pickle=True)

    with open(model_dir / "domain_encoder.pkl", "rb") as f:
        domain_encoder = pickle.load(f)

    from transformers import AutoConfig
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer   = AutoTokenizer.from_pretrained(model_dir)
    bert_config = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased")
    model       = SciBERTWithDomain(config["num_labels"], config["num_domains"], bert_config)
    model.load_state_dict(load_file(model_dir / "model.safetensors"))
    model.to(device).eval()

    return model, tokenizer, topic_classes, domain_encoder, device


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model, tokenizer, topic_classes, domain_encoder, title, abstract, domain, device):
    text     = title + " " + abstract
    encoding = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}

    domain_idx    = int(domain_encoder.transform([domain])[0]) \
                    if domain in domain_encoder.classes_ else 0
    domain_tensor = torch.tensor([domain_idx]).to(device)

    with torch.no_grad():
        logits = model(**encoding, domain=domain_tensor)["logits"][0]

    top3_idx  = logits.topk(3).indices.tolist()
    predicted = topic_classes[top3_idx[0]]
    top3      = [topic_classes[i] for i in top3_idx]
    return predicted, top3


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="scibert_topic_model")
    parser.add_argument("--title",    default="Graph Neural Networks for Spam Detection")
    parser.add_argument("--domain",   default="cs.AI")
    parser.add_argument("--abstract", default=(
        "This paper studies graph-based deep learning for identifying "
        "coordinated spam campaigns in social platforms."
    ))
    args = parser.parse_args()

    model, tokenizer, topic_classes, domain_encoder, device = load_model(args.model_dir)
    predicted, top3 = predict(
        model, tokenizer, topic_classes, domain_encoder,
        args.title, args.abstract, args.domain, device,
    )
    print("\nPredicted Topic:", predicted)
    print("Top 3:", top3)


if __name__ == "__main__":
    main()
