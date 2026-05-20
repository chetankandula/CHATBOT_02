# Research AI Hub

Two AI models in one Streamlit app — **classify** or **generate** research topics from a paper's title, domain, and abstract.

| Model | Task | Output |
|---|---|---|
| SciBERT | Classifies into one of 64 arXiv categories | e.g. `cs.AI`, `math.ST` |
| Flan-T5 | Generates a brand-new research topic | Free-form text |

---

## Folder structure

```
cursor_topic_generator_project/
├── scibert_topic_model/        ← SciBERT trained model (extracted)
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── classes.npy
│   ├── domain_encoder.pkl
│   └── app_config.json
├── scibert_topic_model.zip     ← SciBERT model archive
├── flant5_topic_model/         ← Flan-T5 trained model (extracted)
│   ├── model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── flant5_topic_model.zip      ← Flan-T5 model archive
├── app.py                        ← Streamlit app (home + both chatbots)
├── predict_topic.py              ← SciBERT CLI inference
├── train_topic_generator.py      ← SciBERT training script
├── train_new_topic_generator.py  ← Flan-T5 training script
├── train_and_zip.py              ← Flan-T5 train + auto-zip wrapper
├── topics.json                   ← arXiv code → full name mapping
└── requirements.txt
```

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

> If you get a script execution error, run this once in PowerShell:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## Run the app

```powershell
streamlit run app.py
```

Opens at `http://localhost:8501`. Home page shows two cards — click one to open that model's chatbot.

---

## How it works

**Both chatbots take the same 3 inputs:**

```
Step 1 → Title
Step 2 → Domain
Step 3 → Abstract
```

```
SciBERT:
  Title + Abstract → SciBERT (768-dim) → bottleneck (256) ─┐
  Domain           → Embedding (128-dim) ───────────────────┴→ classifier → 1 of 64 topics

Flan-T5:
  Title + Domain + Abstract → prompt → Flan-T5 → new topic text
```

---

## Sample Inputs

> Use these to test both models right away.

---

### Sample 1 — Computer Science / AI

| Field | Value |
|---|---|
| **Title** | Attention Is All You Need |
| **Domain (SciBERT)** | cs.CL |
| **Domain (Flan-T5)** | Natural Language Processing |
| **Abstract** | We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. |

**SciBERT expected:** `cs.CL` — Computation and Language
**Flan-T5 example output:** *"A Transformer-Based Approach for Efficient Sequence-to-Sequence Learning in Neural Machine Translation"*

---

### Sample 2 — Cybersecurity

| Field | Value |
|---|---|
| **Title** | Network Intrusion Detection Using Deep Learning |
| **Domain (SciBERT)** | cs.CR |
| **Domain (Flan-T5)** | Cybersecurity |
| **Abstract** | We present a deep learning framework for detecting network intrusions in real-time. Our model is trained on the KDD Cup dataset and uses a combination of LSTM and CNN layers to classify network traffic into normal and attack categories. The approach achieves 98.6% accuracy while maintaining low false positive rates. |

**SciBERT expected:** `cs.CR` — Cryptography and Security
**Flan-T5 example output:** *"An Explainable AI Framework for Real-Time Network Threat Detection Using Deep Learning"*

---

### Sample 3 — Healthcare / Medical AI

| Field | Value |
|---|---|
| **Title** | Early Detection of Alzheimer's Disease Using MRI and Machine Learning |
| **Domain (SciBERT)** | eess.IV |
| **Domain (Flan-T5)** | Healthcare |
| **Abstract** | We propose a machine learning pipeline for the early detection of Alzheimer's disease from structural MRI scans. Features are extracted using 3D convolutional neural networks and classified using a support vector machine. The model achieves 94% sensitivity and 91% specificity on a dataset of 500 patients. |

**SciBERT expected:** `eess.IV` — Image and Video Processing
**Flan-T5 example output:** *"A Deep Learning Framework for Intelligent Alzheimer's Prediction Using Neuroimaging Data"*

---

### Sample 4 — Natural Language Processing

| Field | Value |
|---|---|
| **Title** | Sentiment Analysis of Twitter Data Using BERT |
| **Domain (SciBERT)** | cs.CL |
| **Domain (Flan-T5)** | NLP |
| **Abstract** | This paper explores fine-tuning BERT for sentiment classification on Twitter data. We evaluate three fine-tuning strategies on a corpus of 1.2 million tweets across five sentiment classes. Our best model achieves a macro F1-score of 0.87, outperforming previous LSTM-based baselines by 6 points. |

**SciBERT expected:** `cs.CL` — Computation and Language
**Flan-T5 example output:** *"Context-Aware Transformer Model for Multi-Class Sentiment Classification in Social Media"*

---

### Sample 5 — Computer Vision

| Field | Value |
|---|---|
| **Title** | Real-Time Object Detection Using YOLO on Edge Devices |
| **Domain (SciBERT)** | cs.CV |
| **Domain (Flan-T5)** | Computer Vision |
| **Abstract** | We present an optimized YOLO-based object detection pipeline for deployment on resource-constrained edge devices. By applying model pruning and quantization, we reduce the model size by 70% with less than 2% drop in mAP. Inference runs at 30 FPS on a Raspberry Pi 4. |

**SciBERT expected:** `cs.CV` — Computer Vision and Pattern Recognition
**Flan-T5 example output:** *"Adaptive Lightweight Deep Learning Framework for Real-Time Visual Detection on Embedded Systems"*

---

### Sample 6 — Mathematics / Statistics

| Field | Value |
|---|---|
| **Title** | Stochastic Gradient Descent on Non-Convex Functions |
| **Domain (SciBERT)** | math.OC |
| **Domain (Flan-T5)** | Mathematics |
| **Abstract** | We analyze the convergence properties of stochastic gradient descent for non-convex optimization problems. We prove that under mild conditions the algorithm converges to a stationary point and provide bounds on the convergence rate depending on the smoothness of the loss function. |

**SciBERT expected:** `math.OC` — Optimization and Control
**Flan-T5 example output:** *"A Data-Driven Approach for Convergence Analysis of Gradient Optimization in Non-Convex Settings"*

---

## Common arXiv domain codes for SciBERT

| Code | Full Name |
|---|---|
| `cs.AI` | Artificial Intelligence |
| `cs.LG` | Machine Learning |
| `cs.CL` | Computation and Language (NLP) |
| `cs.CV` | Computer Vision |
| `cs.CR` | Cryptography and Security |
| `cs.RO` | Robotics |
| `eess.IV` | Image and Video Processing |
| `eess.SP` | Signal Processing |
| `math.ST` | Statistics Theory |
| `math.OC` | Optimization and Control |
| `stat.ML` | Machine Learning (Statistics) |
| `astro-ph` | Astrophysics |
| `cond-mat` | Condensed Matter |
| `gr-qc` | General Relativity and Quantum Cosmology |
| `hep-th` | High Energy Physics — Theory |
| `quant-ph` | Quantum Physics |
