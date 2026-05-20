# predict_title.py

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# folder where your trained model is saved
MODEL_PATH = "new_research_topic_model"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Using:", device)


def predict_title(category, summary):

    prompt = f"""
Generate a meaningful research title.

Category:
{category}

Summary:
{summary}

Research Title:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=10,
        do_sample=True,
        temperature=1.2,
        top_k=100,
        top_p=0.95,
        repetition_penalty=3.5,
        no_repeat_ngram_size=2
    )

    title = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return title


# User input

category = input("\nEnter Category:\n")
summary = input("\nEnter Summary:\n")

title = predict_title(category, summary)

print("\n======================")
print("GENERATED TITLE")
print("======================")
print(title)