# 🗺️ Penumbra

> The first AI that maps exactly where it's confident vs guessing.

Finetuned Ministral 3B to produce structured **penumbra** alongside every answer, revealing which claims are solid, which are contested and which are guesses.

---

## 🎯 The Problem

Every AI model outputs answers with the same confident tone whether it's certain or guessing.

**Base model:** "The 2008 financial crisis was caused by subprime mortgage lending, deregulation, and CDOs..."

*Everything sounds equally certain. Nothing tells you what to trust.*

**Penumbra:** 
- Subprime lending as trigger: 🟢 95% confident (overwhelming evidence)
- Deregulation as direct cause: 🟡 61% confident (economists debate causality vs correlation)  
- CDOs as primary mechanism: 🟡 73% confident (strong evidence, alternatives exist)

*Now you know exactly what to trust and what to verify.*

---

## 🧠 How It Works

```
Public Datasets          Mistral Large 3
(TruthfulQA,         →   annotates each QA    →   Finetuning data
 TriviaQA, NQ)            with penumbra

                    QLoRA Finetuning
                    Ministral 3B learns to
                    ALWAYS output uncertainty
                    map alongside answer

                    Streamlit Demo
                    Side-by-side: base vs
                    finetuned — judges see
                    the difference instantly
```

---

## 🚀 Quick Start

### 1. Setup
```bash
git clone <repo>
cd penumbra
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Collect Data
```bash
python src/data/collect.py
```

### 3. Annotate with Mistral Large 3
```bash
python src/data/annotate.py
```

### 4. Generate Synthetic Data
```bash
python src/data/generate.py
```

### 5. Prepare Training Data
```bash
python src/data/prepare.py
```

### 6. Finetune Ministral 3B
```bash
python src/training/train.py
```

### 7. Evaluate
```bash
python src/training/evaluate.py
```

### 8. Run Demo
```bash
streamlit run src/app/streamlit_app.py
```

---

## 📊 Model Stack

| Component | Model | Purpose |
|---|---|---|
| Annotator | Mistral Large 3 | Labels training data with penumbra |
| Finetuned | Ministral 3B + QLoRA | Learns to output penumbra |
| Comparison | Ministral 3B (base) | Shows before/after delta |
| Tracking | W&B | Training curves, calibration metrics |

---

## 📈 Evaluation

Three key metrics:

1. **Calibration** — When model says 0.9 confidence, is it right 90% of the time?
2. **Human Agreement** — Do humans agree with uncertainty assessments?
3. **Before/After** — How dramatically does finetuning improve uncertainty awareness?

---

## 🏗️ Project Structure

```
penumbra/
├── src/
│   ├── data_prep/
│   │   ├── collect.py          # Pull TruthfulQA, TriviaQA, NQ, FEVER
│   │   ├── annotate.py         # Mistral Large 3 annotation pipeline
│   │   ├── generate.py         # Synthetic data generation
│   │   └── prepare.py          # Format for finetuning
│   ├── training/
│   │   ├── config.py           # QLoRA hyperparameters
│   │   ├── train.py            # Finetuning loop
│   │   └── evaluate.py         # Calibration + before/after
│   ├── inference/
│   │   └── predict.py          # Penumbra inference
│   └── app/
│       └── streamlit_app.py    # Demo UI
├── data/
│   ├── raw/                    # Downloaded datasets
│   ├── synthetic/              # Annotated + generated data
│   └── processed/              # Train/val/test splits
├── models/                     # Saved finetuned model
├── results/                    # Evaluation outputs
├── .env.example
└── requirements.txt
```

---

## 💡 Why This Matters

- **For users:** Know what to trust vs verify in any AI response
- **For enterprises:** Catch hallucinations before they cause damage  
- **For researchers:** Ground truth signal for epistemic calibration work
- **For AI safety:** Transparency about model uncertainty is a prerequisite for trust

---

Built for the Mistral Hackathon 2026.
