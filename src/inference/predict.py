"""
predict.py — Generate uncertainty maps from finetuned Ministral 3B.
Used by the Streamlit app for inference.
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

MODEL_DIR = Path("models/penumbra-ministral-3b")
HF_REPO_ID = os.getenv("HF_REPO_ID", "Vaaruni2797/penumbra-ministral-3b")
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

SYSTEM_PROMPT = """You are an epistemically transparent AI assistant.
When answering questions, you ALWAYS provide a complete answer AND a structured 
uncertainty map breaking down your confidence in each claim."""


class UncertaintyMapPredictor:
    
    def __init__(self, use_local: bool = False, use_hub: bool = True):
        """
        use_hub:   Load from HuggingFace Hub (default) — 
                   judges can use without any local setup.
        use_local: Load from local disk after training.
        Neither:   Fall back to Mistral API with few-shot prompting.
        """
        self.use_local = use_local
        self.use_hub = use_hub
        self.model = None
        self.tokenizer = None
        
        if use_hub:
            self._load_from_hub()
        elif use_local and MODEL_DIR.exists():
            self._load_local_model()

    def _load_from_hub(self):
        print(f"Loading Penumbra from HuggingFace Hub: {HF_REPO_ID}")
        try:
            base = AutoModelForCausalLM.from_pretrained(
                "mistralai/Ministral-3B-2410",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=os.getenv("HF_TOKEN")
            )
            self.model = PeftModel.from_pretrained(
                base,
                HF_REPO_ID,
                token=os.getenv("HF_TOKEN")
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                HF_REPO_ID,
                token=os.getenv("HF_TOKEN")
            )
            self.model.eval()
            print("✅ Penumbra loaded from Hub.")
        except Exception as e:
            print(f"Hub loading failed: {e}. Falling back to API.")
            self.use_hub = False

    def _load_local_model(self):
        print("Loading local finetuned model...")
        base = AutoModelForCausalLM.from_pretrained(
            "mistralai/Ministral-3B-2410",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, str(MODEL_DIR))
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model.eval()
        print("Model loaded.")

    def predict(self, question: str) -> dict:
        if self.model:
            return self._predict_local(question)
        else:
            return self._predict_api(question)
    
    def _predict_local(self, question: str) -> dict:
        """Generate from local finetuned model."""
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
        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        try:
            return json.loads(generated)
        except json.JSONDecodeError:
            return {"error": "Parse failed", "raw": generated}
    
    def _predict_api(self, question: str) -> dict:
        few_shot = """Example:
Question: What is the boiling point of water?
Response: {
  "answer": "Water boils at 100°C (212°F) at standard atmospheric pressure (1 atm).",
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
  "least_certain_claim": "Water boils at 100°C at standard pressure",
  "epistemic_summary": "This is a physical constant with near-certainty."
}"""
        
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": few_shot},
                {"role": "user", "content": f"Now answer with uncertainty map:\n{question}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}
    
    def get_base_response(self, question: str) -> str:
        response = client.chat.complete(
            model="ministral-3b-2410",
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content


_predictor = None

def get_predictor(use_local: bool = True, use_hub: bool = True) -> UncertaintyMapPredictor:
    global _predictor
    if _predictor is None:
        _predictor = UncertaintyMapPredictor(use_local=use_local, use_hub=use_hub)
    return _predictor
