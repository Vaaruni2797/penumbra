"""
upload_to_hub.py — Upload finetuned Penumbra model to HuggingFace Hub.

Run after training is complete:
    python src/training/upload_to_hub.py

Your model will be live at:
    https://huggingface.co/Vaaruni2797/penumbra-ministral-8b
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME", "Vaaruni2797")
REPO_NAME = "penumbra-ministral-8b"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
MODEL_DIR = Path("models/penumbra-ministral-8b")

MODEL_CARD = """---
language:
- en
license: apache-2.0
base_model: mistralai/Ministral-8B-Instruct-2410
tags:
- uncertainty-quantification
- epistemic-transparency
- finetuned
- qlora
- mistral
- penumbra
---

# 🌒 Penumbra — Uncertainty Maps for Language Models

> Between knowing and guessing, there's a shadow. Penumbra makes it visible.

Penumbra is a finetuned version of [Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)
trained to produce **structured uncertainty maps** alongside every answer.

Unlike standard models that output answers with uniform confident tone,
Penumbra maps exactly which claims are solid, contested, or genuinely uncertain.

---

## 🎯 What It Does

**Standard model:**
> "The 2008 financial crisis was caused by subprime mortgage lending, deregulation, and CDOs..."

*Everything sounds equally certain.*

**Penumbra:**
```json
{
  "answer": "The 2008 financial crisis was caused by...",
  "claims": [
    {
      "claim": "Subprime mortgage lending was a primary trigger",
      "confidence": 0.95,
      "basis": "Overwhelming documented evidence, broad consensus",
      "evidence_quality": "strong",
      "alternative_views": null
    },
    {
      "claim": "Deregulation was a direct cause",
      "confidence": 0.61,
      "basis": "Contested — economists debate causality vs correlation",
      "evidence_quality": "moderate",
      "alternative_views": "Some argue deregulation enabled but did not cause the crisis"
    }
  ],
  "overall_confidence": 0.78,
  "least_certain_claim": "Deregulation was a direct cause",
  "epistemic_summary": "Core facts are well-established; causal attributions remain debated."
}
```

*Now you know exactly what to trust and what to verify.*

---

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Ministral-8B-Instruct-2410",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "Vaaruni2797/penumbra-ministral-8b")
tokenizer = AutoTokenizer.from_pretrained("Vaaruni2797/penumbra-ministral-8b")

# System prompt — required for JSON output
system = \"\"\"You are an epistemically transparent AI assistant.
When answering questions, respond ONLY with valid JSON in exactly this format:
{
  "answer": "complete natural language answer",
  "claims": [
    {
      "claim": "specific assertion",
      "confidence": 0.85,
      "basis": "why this confidence level",
      "evidence_quality": "strong",
      "alternative_views": null
    }
  ],
  "overall_confidence": 0.85,
  "least_certain_claim": "lowest confidence claim",
  "epistemic_summary": "one sentence about overall certainty"
}\"\"\"

question = "Is nuclear energy safe?"
prompt = f"<s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{question} [/INST]"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1, do_sample=True)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

uncertainty_map = json.loads(response)
print(json.dumps(uncertainty_map, indent=2))
```

---

## 🧠 Training Details

| Component | Detail |
|---|---|
| Base model | mistralai/Ministral-8B-Instruct-2410 |
| Method | QLoRA (r=4, alpha=8) |
| Annotator | Mistral Large 3 |
| Training data | TruthfulQA + TriviaQA + FEVER + Synthetic |
| Epochs | 1 |
| Hardware | RTX 3070 8GB |
| Token accuracy | 82.8% |

---

## 📊 Live Demo

Try it at: [HuggingFace Spaces — Penumbra](https://huggingface.co/spaces/Vaaruni2797/penumbra)

---

Built for the Mistral Hackathon 2026. 🌒
"""


def upload_adapter_weights():
    """Upload QLoRA adapter weights to HuggingFace Hub."""

    api = HfApi(token=os.getenv("HF_TOKEN"))

    print(f"Creating repo: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN"),
            private=False,
            exist_ok=True
        )
        print(f"  ✅ Repo ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"  Repo may already exist: {e}")

    # Upload adapter files
    print("\nUploading adapter weights...")
    adapter_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]

    for filename in adapter_files:
        filepath = MODEL_DIR / filename
        if filepath.exists():
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=REPO_ID,
                token=os.getenv("HF_TOKEN")
            )
            print(f"  ✅ Uploaded {filename}")
        else:
            print(f"  ⚠️  Skipped {filename} (not found)")

    # Upload model card
    print("\nUploading model card...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        token=os.getenv("HF_TOKEN")
    )
    print("  ✅ Model card uploaded")

    # Upload training config
    config_data = {
        "base_model": "mistralai/Ministral-8B-Instruct-2410",
        "method": "QLoRA",
        "lora_r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
        "training_data": ["TruthfulQA", "TriviaQA", "FEVER", "Synthetic"],
        "epochs": 1,
        "learning_rate": 2e-4,
        "token_accuracy": 0.828,
        "project": "Penumbra — Uncertainty Maps"
    }
    api.upload_file(
        path_or_fileobj=json.dumps(config_data, indent=2).encode(),
        path_in_repo="training_config.json",
        repo_id=REPO_ID,
        token=os.getenv("HF_TOKEN")
    )
    print("  ✅ Training config uploaded")

    print(f"\n🌒 Penumbra is live!")
    print(f"   Model: https://huggingface.co/{REPO_ID}")
    print(f"   Share this with judges!")

    return f"https://huggingface.co/{REPO_ID}"


def verify_upload():
    """Sanity check — load model from Hub and run inference."""
    print("\nVerifying upload by loading from Hub...")
    try:
        base = AutoModelForCausalLM.from_pretrained(
            "mistralai/Ministral-8B-Instruct-2410",
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            token=os.getenv("HF_TOKEN")
        )
        model = PeftModel.from_pretrained(base, REPO_ID, device_map={"": 0})
        tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

        system = """You are an epistemically transparent AI assistant.
When answering questions, respond ONLY with valid JSON containing answer, claims, and confidence scores."""

        question = "What is the speed of light?"
        prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{question} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        result = json.loads(response)
        print(f"  ✅ Model loads and runs correctly from Hub")
        print(f"  overall_confidence: {result.get('overall_confidence')}")
        return True

    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("🌒 UPLOADING PENUMBRA TO HUGGINGFACE HUB")
    print("=" * 50)

    model_url = upload_adapter_weights()
    verify_upload()

    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("=" * 50)
    print("1. Deploy Streamlit app to HF Spaces:")
    print("   huggingface-cli repo create penumbra --type space --space-sdk streamlit")
    print("2. Push your app:")
    print("   git push https://huggingface.co/spaces/Vaaruni2797/penumbra main")
    print("3. Share the live URL with judges:")
    print(f"   https://huggingface.co/spaces/Vaaruni2797/penumbra")