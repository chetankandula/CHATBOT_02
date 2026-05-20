# evaluate_flan_t5.py

import pandas as pd
import torch
import  evaluate
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA_PATH = "arXiv_scientific dataset.csv"
MODEL_PATH = "flant5_category_summary_topic_model"

df = pd.read_csv(DATA_PATH)
df = df[["category", "summary", "title"]].dropna()

_, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

predictions = []
references = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    category = row["category"]
    summary = row["summary"]
    actual_title = row["title"]

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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,
            repetition_penalty=2.5,
            no_repeat_ngram_size=2
        )

    generated_title = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predictions.append(generated_title)
    references.append(actual_title)

rouge_scores = rouge.compute(
    predictions=predictions,
    references=references
)

bleu_score = bleu.compute(
    predictions=predictions,
    references=[[r] for r in references]
)

pred_emb = semantic_model.encode(predictions, convert_to_tensor=True)
ref_emb = semantic_model.encode(references, convert_to_tensor=True)

similarities = util.cos_sim(pred_emb, ref_emb).diagonal()
semantic_similarity = similarities.mean().item()

print("\n==============================")
print("FLAN-T5 GENERATION EVALUATION")
print("==============================")
print("ROUGE-1:", rouge_scores["rouge1"])
print("ROUGE-2:", rouge_scores["rouge2"])
print("ROUGE-L:", rouge_scores["rougeL"])
print("BLEU:", bleu_score["bleu"])
print("Semantic Similarity:", semantic_similarity)

results_df = pd.DataFrame({
    "category": test_df["category"].tolist(),
    "summary": test_df["summary"].tolist(),
    "actual_title": references,
    "generated_title": predictions
})

results_df.to_csv("flant5_evaluation_predictions.csv", index=False)

print("\nSaved predictions to flant5_evaluation_predictions.csv")