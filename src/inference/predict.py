"""
predict.py — Generate uncertainty maps from finetuned Ministral 8B.
Used by the Streamlit app for inference.

Priority order (controlled by use_local / use_hub / PENUMBRA_API_URL):
  1. Modal API endpoint  — if PENUMBRA_API_URL is set in .env
  2. HuggingFace Hub     — loads adapter weights from Hub onto local GPU
  3. Local disk          — loads from models/ directory after training
  4. Mistral API         — few-shot fallback, no GPU required
"""

import json
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR   = Path("models/penumbra-ministral-8b")
HF_REPO_ID  = os.getenv("HF_REPO_ID", "Vaaruni2797/penumbra-ministral-8b")
MODAL_URL   = os.getenv("PENUMBRA_API_URL", "")   # set this after `modal deploy`

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

SYSTEM_PROMPT = """You are an epistemically transparent AI assistant.
When answering questions, you ALWAYS provide a complete answer AND a structured
uncertainty map breaking down your confidence in each claim."""


class UncertaintyMapPredictor:

    def __init__(self, use_local: bool = False, use_hub: bool = True):
        """
        Resolution order:
          PENUMBRA_API_URL set  →  Modal cloud endpoint (no GPU needed locally)
          use_hub=True          →  HuggingFace Hub adapter
          use_local=True        →  Local disk model
          fallback              →  Mistral large API (few-shot)
        """
        self.use_local = use_local
        self.use_hub   = use_hub
        self.model     = None
        self.tokenizer = None
        self.modal_url = MODAL_URL.rstrip("/") if MODAL_URL else None

        if self.modal_url:
            print(f"✅ Using Modal API endpoint: {self.modal_url}")
        elif use_hub:
            self._load_from_hub()
        elif use_local and MODEL_DIR.exists():
            self._load_local_model()
        else:
            print("No local model or Modal URL — using Mistral API fallback.")

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_from_hub(self):
        print(f"Loading Penumbra from HuggingFace Hub: {HF_REPO_ID}")
        try:
            base = AutoModelForCausalLM.from_pretrained(
                "mistralai/Ministral-8B-Instruct-2410",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=os.getenv("HF_TOKEN"),
            )
            self.model = PeftModel.from_pretrained(
                base, HF_REPO_ID, token=os.getenv("HF_TOKEN")
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                HF_REPO_ID, token=os.getenv("HF_TOKEN")
            )
            self.model.eval()
            print("✅ Penumbra loaded from Hub.")
        except Exception as e:
            print(f"Hub loading failed: {e}. Falling back to API.")
            self.use_hub = False

    def _load_local_model(self):
        print("Loading local finetuned model…")
        base = AutoModelForCausalLM.from_pretrained(
            "mistralai/Ministral-8B-Instruct-2410",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model     = PeftModel.from_pretrained(base, str(MODEL_DIR))
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model.eval()
        print("✅ Local model loaded.")

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(self, question: str) -> dict:
        if self.modal_url:
            return self._predict_modal(question)
        if self.model:
            return self._predict_local(question)
        return self._predict_api(question)

    def get_base_response(self, question: str) -> str:
        if self.modal_url:
            return self._base_modal(question)
        return self._base_api(question)

    # ── Modal endpoint ───────────────────────────────────────────────────────

    def _predict_modal(self, question: str) -> dict:
        """Call the deployed Modal endpoint — no GPU required locally."""
        import requests
        try:
            r = requests.post(
                self.modal_url,
                json={"question": question, "mode": "penumbra"},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Modal call failed: {e}. Falling back to Mistral API.")
            return self._predict_api(question)

    def _base_modal(self, question: str) -> str:
        import requests
        try:
            r = requests.post(
                self.modal_url,
                json={"question": question, "mode": "base"},
                timeout=60,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            print(f"Modal base call failed: {e}. Falling back to Mistral API.")
            return self._base_api(question)

    # ── Local model ──────────────────────────────────────────────────────────

    def _predict_local(self, question: str) -> dict:
        system = """You are an epistemically transparent AI assistant.
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
}"""
        prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{question} [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        try:
            return json.loads(generated)
        except json.JSONDecodeError:
            return {"error": "Parse failed", "raw": generated}

    # ── Mistral API fallback ─────────────────────────────────────────────────

    def _predict_api(self, question: str) -> dict:
        few_shot = """Example:
Question: What is the boiling point of water?
Response: {
  "answer": "Water boils at 100°C (212°F) at standard atmospheric pressure.",
  "claims": [
    {
      "claim": "Water boils at 100°C at standard pressure",
      "confidence": 0.99,
      "basis": "Fundamental physical constant, verified countless times",
      "evidence_quality": "strong",
      "alternative_views": null
    },
    {
      "claim": "Boiling point changes with altitude/pressure",
      "confidence": 0.97,
      "basis": "Well-established physics — lower pressure = lower boiling point",
      "evidence_quality": "strong",
      "alternative_views": null
    }
  ],
  "overall_confidence": 0.98,
  "least_certain_claim": "Boiling point changes with altitude/pressure",
  "epistemic_summary": "This is a physical constant with near-certainty."
}"""
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": few_shot},
                {"role": "user",    "content": f"Now answer with uncertainty map:\n{question}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        try:
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

    def _base_api(self, question: str) -> str:
        response = client.chat.complete(
            model="ministral-8b-latest",
            messages=[{"role": "user", "content": question}],
        )
        return response.choices[0].message.content


# ─── Singleton ───────────────────────────────────────────────────────────────

_predictor = None

def get_predictor(use_local: bool = True, use_hub: bool = True) -> UncertaintyMapPredictor:
    global _predictor
    if _predictor is None:
        _predictor = UncertaintyMapPredictor(use_local=use_local, use_hub=use_hub)
    return _predictor