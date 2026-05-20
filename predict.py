from pathlib import Path
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


MODEL_PATH = r"C:\Users\windows\Downloads\cursor_topic_generator_project\new_research_topic_model"

model_path = Path(MODEL_PATH)

print("Loading model from:", model_path)

tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), local_files_only=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Using device:", device)


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_keywords(summary, max_keywords=8):
    text = summary.lower()
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)

    custom_stopwords = {
        "research", "study", "using", "system", "systems",
        "model", "models", "framework", "analysis", "based"
    }

    words = [
        w for w in words
        if w not in ENGLISH_STOP_WORDS and w not in custom_stopwords
    ]

    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    return [w for w, c in sorted_words[:max_keywords]]


def is_too_similar(title, output):
    title = title.lower().strip()
    output = output.lower().strip()

    title_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", title))
    output_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", output))

    if title == output:
        return True

    if title in output:
        return True

    if len(title_words) > 0:
        overlap = len(title_words.intersection(output_words)) / len(title_words)
        if overlap >= 0.6:
            return True

    return False


def fallback_topic(title, domain, summary):
    keywords = extract_keywords(summary)

    domain_lower = domain.lower()

    if "cyber" in domain_lower or "security" in domain_lower:
        if "explainable" in summary.lower() or "ai" in summary.lower():
            return "Explainable Adaptive Cyber Threat Detection Using Streaming Machine Learning"
        return "Adaptive Cyber Threat Detection Framework for Dynamic Network Traffic"

    if "nlp" in domain_lower or "natural language" in domain_lower:
        return "Context-Aware Deep Learning Framework for Intelligent Text Understanding"

    if "health" in domain_lower or "medical" in domain_lower:
        return "Deep Learning Framework for Intelligent Healthcare Prediction and Decision Support"

    if "vision" in domain_lower or "image" in domain_lower:
        return "Transformer-Based Computer Vision Framework for Real-Time Visual Intelligence"

    if len(keywords) >= 4:
        phrase = " ".join(keywords[:4]).title()
        return f"Intelligent {domain} Framework for {phrase}"

    return f"Advanced {domain} Framework Using Deep Learning and Intelligent Analytics"


def build_prompt(title, domain, summary):
    return f"""
Create a new research topic from the domain and summary.

Do not copy this title: {title}

Domain:
{domain}

Summary:
{summary}

New topic:
"""


def predict_topic(title, domain, summary):
    prompt = build_prompt(title, domain, summary)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        num_beams=10,
        do_sample=True,
        temperature=1.4,
        top_k=100,
        top_p=0.95,
        repetition_penalty=4.0,
        no_repeat_ngram_size=2
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = clean_text(generated)

    if is_too_similar(title, generated):
        generated = fallback_topic(title, domain, summary)

    return generated


title = input("\nEnter Existing Title:\n")
domain = input("\nEnter Research Domain:\n")
summary = input("\nEnter Research Summary:\n")

topic = predict_topic(title, domain, summary)

print("\n==============================")
print("GENERATED RESEARCH TOPIC")
print("==============================\n")
print(topic)