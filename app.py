"""
Research AI Hub — Streamlit App
Home page → SciBERT classifier OR Flan-T5 generator chatbot.

Run:  streamlit run app.py
"""

import json
import pickle
import re
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Research AI Hub", page_icon="🔬", layout="centered")

st.markdown("""
<style>
/* ── cards on home page ── */
.model-card {
    border-radius: 16px;
    padding: 28px 24px;
    color: white;
    margin-bottom: 12px;
    min-height: 180px;
}
.card-scibert { background: linear-gradient(135deg, #1e3a5f 0%, #0d6efd 100%); }
.card-flant5  { background: linear-gradient(135deg, #1a3a2a 0%, #1a7a3a 100%); }
.model-card h2 { margin: 0 0 6px 0; font-size: 22px; }
.model-card p  { margin: 0 0 18px 0; font-size: 14px; opacity: 0.85; line-height: 1.5; }
.model-card .tag {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border-radius: 20px; padding: 3px 12px;
    font-size: 12px; margin-right: 6px; margin-bottom: 10px;
}
/* ── result card ── */
.topic-card {
    border-radius: 14px; padding: 20px 24px; margin: 8px 0; color: white;
}
.topic-scibert { background: linear-gradient(135deg, #1e3a5f 0%, #0d6efd 100%); }
.topic-flant5  { background: linear-gradient(135deg, #1a3a2a 0%, #1a7a3a 100%); }
.topic-card h3 { margin: 0 0 4px 0; font-size: 12px; letter-spacing: 1px; opacity: 0.75; }
.topic-card p  { margin: 0; font-size: 22px; font-weight: 700; line-height: 1.4; }
.badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    border-radius: 20px; padding: 3px 12px; font-size: 12px;
    margin-top: 10px; margin-right: 6px;
}
.conf-wrap { margin-top: 14px; }
.conf-label { font-size: 11px; letter-spacing: 0.8px; text-transform: uppercase; opacity: 0.55; margin-bottom: 6px; }
.conf-track { height: 8px; background: rgba(255,255,255,0.12); border-radius: 99px; overflow: hidden; }
.conf-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg,
        rgba(255,255,255,0.35) 0%, rgba(255,255,255,0.95) 45%,
        rgba(255,255,255,1) 50%, rgba(255,255,255,0.95) 55%,
        rgba(255,255,255,0.35) 100%);
    background-size: 200% 100%;
    animation: shine 2.2s ease-in-out infinite;
}
@keyframes shine {
    0%   { background-position: 200% center; }
    100% { background-position: -200% center; }
}
.conf-pct { font-size: 12px; opacity: 0.7; margin-top: 5px; text-align: right; }
</style>
""", unsafe_allow_html=True)


# ── paths ─────────────────────────────────────────────────────────────────────

SCIBERT_DIR = Path("scibert_topic_model")
FLANT5_DIR  = Path("flant5_topic_model")

with open(Path(__file__).parent / "topics.json", encoding="utf-8") as _f:
    ARXIV_NAMES: dict[str, str] = json.load(_f)

def full_name(code: str) -> str:
    return ARXIV_NAMES.get(code, code)


# ── model loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading SciBERT…")
def load_scibert():
    import numpy as np
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    class SciBERTWithDomain(nn.Module):
        def __init__(self, num_labels, num_domains, bert_config):
            super().__init__()
            self.bert             = AutoModel.from_config(bert_config)
            self.domain_embedding = nn.Embedding(num_domains, 128)
            self.text_bottleneck  = nn.Sequential(nn.Linear(768, 256))
            self.classifier       = nn.Sequential(
                nn.Linear(384, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, num_labels),
            )

        def forward(self, input_ids, attention_mask, domain, **kwargs):
            cls   = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            text  = self.text_bottleneck(cls)
            dom   = self.domain_embedding(domain)
            return {"logits": self.classifier(torch.cat((text, dom), dim=1))}

    with open(SCIBERT_DIR / "app_config.json") as f:
        cfg = json.load(f)
    with open(SCIBERT_DIR / "domain_encoder.pkl", "rb") as f:
        domain_enc = pickle.load(f)

    classes    = np.load(SCIBERT_DIR / "classes.npy", allow_pickle=True)
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer  = AutoTokenizer.from_pretrained(SCIBERT_DIR)
    bert_cfg   = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased")
    model      = SciBERTWithDomain(cfg["num_labels"], cfg["num_domains"], bert_cfg)
    model.load_state_dict(load_file(SCIBERT_DIR / "model.safetensors"))
    model.to(device).eval()
    return model, tokenizer, classes, domain_enc, device


@st.cache_resource(show_spinner="Loading Flan-T5…")
def load_flant5():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(FLANT5_DIR))
    model     = AutoModelForSeq2SeqLM.from_pretrained(str(FLANT5_DIR))
    model.to(device).eval()
    return model, tokenizer, device


# ── inference helpers ─────────────────────────────────────────────────────────

def predict_scibert(model, tokenizer, classes, domain_enc, device, title, abstract, domain=""):
    import torch
    import torch.nn.functional as F

    text = title + " " + abstract
    enc  = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    enc  = {k: v.to(device) for k, v in enc.items()}

    dom_idx = int(domain_enc.transform([domain])[0]) if domain in domain_enc.classes_ else 0
    dom_t   = torch.tensor([dom_idx]).to(device)

    with torch.no_grad():
        logits = model(**enc, domain=dom_t)["logits"][0]

    probs    = F.softmax(logits, dim=0)
    top3_idx = logits.topk(3).indices.tolist()
    predicted   = str(classes[top3_idx[0]])
    top3        = [str(classes[i]) for i in top3_idx]
    confidence  = round(probs[top3_idx[0]].item() * 100, 1)
    return predicted, top3, confidence


def _clean(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _too_similar(title, output):
    t, o = title.lower().strip(), output.lower().strip()
    if t == o or t in o:
        return True
    tw = set(re.findall(r"\b[a-zA-Z]{4,}\b", t))
    ow = set(re.findall(r"\b[a-zA-Z]{4,}\b", o))
    return bool(tw) and len(tw & ow) / len(tw) >= 0.6


def generate_flant5(model, tokenizer, device, title, domain, summary):
    import torch

    prompt = _clean(f"""
Create a new research topic from the domain and summary.
Do not copy this title: {title}
Domain: {domain}
Summary: {summary}
New topic:
""")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=48, num_beams=8,
            do_sample=True, temperature=1.2, top_k=80, top_p=0.95,
            repetition_penalty=3.5, no_repeat_ngram_size=3, early_stopping=True,
        )

    result = _clean(tokenizer.decode(outputs[0], skip_special_tokens=True))

    if not result or _too_similar(title, result):
        kw = re.findall(r"\b[A-Za-z]{5,}\b", summary)[:4]
        phrase = " ".join(kw).title() if kw else "Intelligent Systems"
        result = f"A Deep Learning Framework for {phrase} in {domain}"

    return result


# ── state helpers ─────────────────────────────────────────────────────────────

def init():
    defaults = {"page": "home", "messages": [], "step": None, "inputs": {}, "greeted": False}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def go(page):
    st.session_state.page     = page
    st.session_state.messages = []
    st.session_state.step     = "title"
    st.session_state.inputs   = {}
    st.session_state.greeted  = False

def bot_says(text, html=False):
    st.session_state.messages.append({"role": "assistant", "content": text, "html": html})

def user_says(text):
    st.session_state.messages.append({"role": "user", "content": text})


# ── pages ─────────────────────────────────────────────────────────────────────

def page_home():
    st.title("🔬 Research AI Hub")
    st.markdown("Choose a model to get started.")
    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="model-card card-scibert">
            <h2>🔵 SciBERT</h2>
            <p>Classifies your paper into one of <b>64 arXiv research categories</b> using title and abstract.</p>
            <span class="tag">Classifier</span>
            <span class="tag">64 topics</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Use SciBERT →", use_container_width=True, key="btn_scibert"):
            go("scibert")
            st.rerun()

    with col2:
        st.markdown("""
        <div class="model-card card-flant5">
            <h2>🟢 Flan-T5</h2>
            <p>Generates a <b>completely new research topic</b> from your domain and summary using Flan-T5.</p>
            <span class="tag">Generator</span>
            <span class="tag">Free-form</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Use Flan-T5 →", use_container_width=True, key="btn_flant5"):
            go("flant5")
            st.rerun()


def page_scibert():
    if not SCIBERT_DIR.exists():
        st.error("`scibert_topic_model/` folder not found.")
        st.stop()

    model, tokenizer, classes, domain_enc, device = load_scibert()

    PROMPTS = {
        "title":    "**Step 1 / 3** — What is the **title** of the paper?",
        "domain":   "**Step 2 / 3** — What is the **research domain**? (e.g. cs.AI, cs.LG, math.ST, eess.IV)",
        "abstract": "**Step 3 / 3** — Paste the **abstract** of the paper.",
    }

    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Back"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        st.title("🔵 SciBERT Topic Classifier")
    st.caption("SciBERT · 64 arXiv topic classes · title + domain + abstract")

    if not st.session_state.greeted:
        bot_says("👋 Hi! Give me a paper's **title**, **domain**, and **abstract** and I'll predict its research topic.\n\n" + PROMPTS["title"])
        st.session_state.greeted = True

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"], unsafe_allow_html=msg.get("html", False))

    step = st.session_state.step
    placeholders = {
        "title":    "e.g. Graph Neural Networks for Spam Detection…",
        "domain":   "e.g. cs.AI, cs.LG, math.ST, eess.IV…",
        "abstract": "Paste the abstract here…",
        "done":     "Type 'again' for another prediction…",
    }
    user_input = st.chat_input(placeholders.get(step, "Type here…"))

    if user_input:
        user_input = user_input.strip()

        if step == "done":
            user_says(user_input)
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            if any(w in user_input.lower() for w in ("again", "new", "another")):
                st.session_state.step    = "title"
                st.session_state.inputs  = {}
                st.session_state.greeted = False
                bot_says("Sure! Let's go again.\n\n" + PROMPTS["title"])
            else:
                bot_says("Type **again** to predict a new topic!")
            st.rerun()
            return

        user_says(user_input)
        st.session_state.inputs[step] = user_input
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        step_order = ["title", "domain", "abstract"]
        idx = step_order.index(step)
        next_step = step_order[idx + 1] if idx + 1 < len(step_order) else "predict"

        if next_step in PROMPTS:
            st.session_state.step = next_step
            bot_says(PROMPTS[next_step])
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(PROMPTS[next_step])
        else:
            st.session_state.step = "done"
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Running SciBERT inference…"):
                    predicted, top3, confidence = predict_scibert(
                        model, tokenizer, classes, domain_enc, device,
                        title    = st.session_state.inputs["title"],
                        abstract = st.session_state.inputs["abstract"],
                        domain   = st.session_state.inputs["domain"],
                    )
                badges = "".join(f'<span class="badge" title="{t}">{full_name(t)}</span>' for t in top3)
                card = f"""
                <div class="topic-card topic-scibert">
                    <h3>PREDICTED RESEARCH TOPIC</h3>
                    <p>{full_name(predicted)}</p>
                    <div style="font-size:13px;opacity:0.6;margin-bottom:8px">{predicted}</div>
                    <div class="conf-wrap">
                        <div class="conf-label">Confidence</div>
                        <div class="conf-track"><div class="conf-fill" style="width:{confidence}%"></div></div>
                        <div class="conf-pct">{confidence}%</div>
                    </div>
                    {badges}
                </div>"""
                st.markdown(card, unsafe_allow_html=True)
                follow = "\n\n✅ Done! Type **again** to predict another."
                st.markdown(follow)
            st.session_state.messages.append({"role": "assistant", "content": card + follow, "html": True})

        st.rerun()


def page_flant5():
    if not FLANT5_DIR.exists():
        st.error("`flant5_topic_model/` folder not found. Unzip the model here first.")
        st.stop()

    model, tokenizer, device = load_flant5()

    PROMPTS = {
        "title":   "**Step 1 / 3** — What is the **existing paper title**?",
        "domain":  "**Step 2 / 3** — What is the **research domain**? (e.g. NLP, Computer Vision, Healthcare)",
        "summary": "**Step 3 / 3** — Paste the **abstract or summary** of the paper.",
    }

    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Back"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        st.title("🟢 Flan-T5 Topic Generator")
    st.caption("Flan-T5 · Generative model · Trained on arXiv dataset")

    if not st.session_state.greeted:
        bot_says("👋 Hi! Give me a paper's **title**, **domain**, and **abstract** and I'll generate a brand-new research topic.\n\n" + PROMPTS["title"])
        st.session_state.greeted = True

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"], unsafe_allow_html=msg.get("html", False))

    step = st.session_state.step
    placeholders = {
        "title":   "e.g. Graph Neural Networks for Spam Detection…",
        "domain":  "e.g. NLP, Computer Vision, Cybersecurity…",
        "summary": "Paste the abstract here…",
        "done":    "Type 'again' for another topic…",
    }
    user_input = st.chat_input(placeholders.get(step, "Type here…"))

    if user_input:
        user_input = user_input.strip()

        if step == "done":
            user_says(user_input)
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            if any(w in user_input.lower() for w in ("again", "new", "another")):
                st.session_state.step   = "title"
                st.session_state.inputs = {}
                st.session_state.greeted = False
                bot_says("Sure! Let's go again.\n\n" + PROMPTS["title"])
            else:
                bot_says("Type **again** to generate a new topic!")
            st.rerun()
            return

        user_says(user_input)
        st.session_state.inputs[step] = user_input
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        step_order = ["title", "domain", "summary"]
        idx = step_order.index(step)
        next_step = step_order[idx + 1] if idx + 1 < len(step_order) else "predict"

        if next_step in PROMPTS:
            st.session_state.step = next_step
            bot_says(PROMPTS[next_step])
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(PROMPTS[next_step])
        else:
            st.session_state.step = "done"
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Generating new research topic…"):
                    new_topic = generate_flant5(
                        model, tokenizer, device,
                        title   = st.session_state.inputs["title"],
                        domain  = st.session_state.inputs["domain"],
                        summary = st.session_state.inputs["summary"],
                    )
                card = f"""
                <div class="topic-card topic-flant5">
                    <h3>GENERATED RESEARCH TOPIC</h3>
                    <p>{new_topic}</p>
                </div>"""
                st.markdown(card, unsafe_allow_html=True)
                follow = "\n\n✅ Done! Type **again** to generate another."
                st.markdown(follow)
            st.session_state.messages.append({"role": "assistant", "content": card + follow, "html": True})

        st.rerun()


# ── router ────────────────────────────────────────────────────────────────────

def main():
    init()
    page = st.session_state.page
    if page == "home":
        page_home()
    elif page == "scibert":
        page_scibert()
    elif page == "flant5":
        page_flant5()


if __name__ == "__main__":
    main()
