# 🌒 Penumbra

> Between knowing and guessing, there's a shadow. Penumbra makes it visible.

Finetuned Ministral 8B to produce structured **uncertainty maps** alongside every answer — revealing which claims are solid, which are contested, and which are guesses.

---

## 🎯 The Problem

Every AI model outputs answers with the same confident tone — whether it's certain or guessing.

**Base model:** "The 2008 financial crisis was caused by subprime mortgage lending, deregulation, and CDOs..."

*Everything sounds equally certain. Nothing tells you what to trust.*

**Penumbra:**
- Subprime lending as trigger: 🟣 95% confident (overwhelming evidence)
- Deregulation as direct cause: 🔵 61% confident (economists debate causality vs correlation)
- CDOs as primary mechanism: 🔵 73% confident (strong evidence, alternatives exist)

*Now you know exactly what to trust and what to verify.*

---

## 🧠 How It Works

| Step | What happens |
|---|---|
| 1️⃣ **Collect** | Pull TruthfulQA, TriviaQA, FEVER — datasets with known uncertainty levels |
| 2️⃣ **Annotate** | Mistral Large 3 decomposes each answer into claims and assigns confidence scores |
| 3️⃣ **Generate** | Synthetic QA pairs created across high / medium / low certainty domains |
| 4️⃣ **Finetune** | Ministral 8B trained via QLoRA to always output structured uncertainty maps |
| 5️⃣ **Demo** | Side-by-side: base model gives confident prose, Penumbra maps the uncertainty |

---

## 🚀 Quick Start

### 1. Setup
```bash
git clone https://github.com/Vaaruni2797/penumbra
cd penumbra
pip install -r requirements.txt

# Mac/Linux:
cp .env.example .env
# Windows:
copy .env.example .env

# Fill in your API keys in .env
```

### 2. Collect Data
```bash
python src/data_prep/collect.py
```

### 3. Annotate with Mistral Large 3
```bash
python src/data_prep/annotate.py
```

### 4. Generate Synthetic Data
```bash
python src/data_prep/generate.py
```

### 5. Prepare Training Data
```bash
python src/data_prep/prepare.py
```

### 6. Finetune Ministral 8B
```bash
python src/training/train.py
```

### 7. Upload to HuggingFace Hub
```bash
python src/training/upload_to_hub.py
```

### 8. Run Demo
```bash
streamlit run src/app/streamlit_app.py
```

---

## 📊 Model Stack

| Component | Model | Purpose |
|---|---|---|
| Annotator | Mistral Large 3 (`mistral-large-latest`) | Labels training data with uncertainty maps |
| Finetuned | Ministral 8B + QLoRA | Learns to output structured uncertainty maps |
| Comparison | Ministral 8B base (`ministral-8b-latest`) | Shows before/after delta |
| Tracking | W&B | Training curves, loss, token accuracy |

---

## 🧪 Training Details

| Component | Detail |
|---|---|
| Base model | mistralai/Ministral-8B-Instruct-2410 |
| Method | QLoRA (r=4, alpha=8) |
| Target modules | q_proj, v_proj |
| Training data | TruthfulQA + TriviaQA + FEVER + Synthetic (545 examples) |
| Epochs | 1 |
| Batch size | 1 (grad accum 16, effective batch 16) |
| Max sequence length | 512 |
| Hardware | RTX 3070 8GB |
| Training time | ~1 hour |
| Final loss | 0.87 |
| Token accuracy | 82.8% |

---

## 🔑 Required API Keys

| Key | Where to get it | When needed |
|---|---|---|
| `MISTRAL_API_KEY` | console.mistral.ai | Data collection & annotation |
| `WANDB_API_KEY` | wandb.ai/settings | Training |
| `HF_TOKEN` (write) | huggingface.co/settings/tokens | Upload to Hub only |

---

## 🖥️ Demo UI

The Streamlit app shows a side-by-side comparison:

| Penumbra (left) | Base Ministral 8B (right) |
|---|---|
| Overall confidence gauge | Standard prose response |
| Epistemic summary | No uncertainty signal |
| Answer | Confident tone regardless of certainty |
| Weakest claim highlighted | |
| Confidence bar chart (dark purple → yellow) | |
| Claim-by-claim breakdown with color-coded confidence | |
| Stats strip: claims analyzed, avg confidence, uncertain claims | |

**Confidence color scale:** ⚫ dark purple = low confidence → 🟣 purple = medium → 🟡 yellow = high confidence

**Sidebar features:**
- Toggle local model vs API mode
- W&B training run link
- Upload `.txt` or `.jsonl` file for batch analysis (up to 10 questions)

---

## 🏗️ Project Structure

```
penumbra/
├── src/
│   ├── data_prep/
│   │   ├── collect.py          # Pull TruthfulQA, TriviaQA, FEVER
│   │   ├── annotate.py         # Mistral Large 3 annotation pipeline
│   │   ├── generate.py         # Synthetic data generation (1 example per call)
│   │   └── prepare.py          # Format for finetuning (80/10/10 split)
│   ├── training/
│   │   ├── config.py           # QLoRA hyperparameters
│   │   ├── train.py            # SFTTrainer + SFTConfig finetuning loop
│   │   ├── evaluate.py         # Calibration + before/after comparison
│   │   └── upload_to_hub.py    # Push adapter weights to HuggingFace
│   ├── inference/
│   │   └── predict.py          # Load from Hub or local, generate uncertainty maps
│   └── app/
│       └── streamlit_app.py    # Side-by-side demo UI
├── .env.example                # Required API keys template
├── .gitignore
└── requirements.txt
```

---

## 🤗 HuggingFace

- **Model:** https://huggingface.co/Vaaruni2797/penumbra-ministral-8b
- **Demo:** https://huggingface.co/spaces/Vaaruni2797/penumbra

---

## 💡 Why This Matters

- **For users:** Know what to trust vs verify in any AI response
- **For enterprises:** Catch hallucinations before they cause damage
- **For researchers:** Ground truth signal for epistemic calibration work
- **For AI safety:** Transparency about model uncertainty is a prerequisite for trust

---

Built for the Mistral Hackathon 2026. 🌒

---

**Team:** nofreelunch &nbsp;·&nbsp; **Author:** Vaaruni Desai

🌐 **Live Demo:** https://penumbra-ynafj2aazjsy9bpdtrn8a5.streamlit.app/