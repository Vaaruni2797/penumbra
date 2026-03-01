"""
modal_app.py — Deploy Penumbra as a serverless GPU API on Modal.

Setup:
    pip install modal
    modal token new          # authenticate once
    modal deploy modal_app.py  # deploy (creates a permanent endpoint)
    modal serve modal_app.py   # hot-reload for dev

Endpoint:
    POST https://<your-workspace>--penumbra-inference-predict.modal.run
    Body: {"question": "..."}

Set these as Modal secrets (modal.com → Secrets → Create):
    HF_TOKEN        — HuggingFace read token
    MISTRAL_API_KEY — fallback if model fails to load
"""

import modal

# ─── Image: all deps baked in ────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.0",
        "transformers==4.40.0",
        "peft==0.10.0",
        "accelerate==0.29.0",
        "sentencepiece",
        "protobuf",
        "mistralai",
        "python-dotenv",
        "fastapi[standard]",
    )
)

app = modal.App("penumbra-inference", image=image)

# ─── Secrets (set once at modal.com/secrets) ─────────────────────────────────
secrets = [modal.Secret.from_name("penumbra-secrets")]

# ─── Model volume — cache weights so cold starts don't re-download ───────────
volume = modal.Volume.from_name("penumbra-weights", create_if_missing=True)
MOUNT_PATH = "/model-cache"

HF_BASE   = "mistralai/Ministral-8B-Instruct-2410"
HF_ADAPTER = "Vaaruni2797/penumbra-ministral-8b"

SYSTEM_PROMPT = """You are an epistemically transparent AI assistant.
When answering questions, you ALWAYS provide a complete answer AND a structured
uncertainty map breaking down your confidence in each claim."""

# ─── Inference class ─────────────────────────────────────────────────────────
@app.cls(
    gpu="A10G",                    # ~$0.001/sec; swap to "T4" for cheaper cold demos
    timeout=180,
    volumes={MOUNT_PATH: volume},
    secrets=secrets,
    container_idle_timeout=300,    # keep warm for 5 min between requests
)
class PenumbraModel:

    @modal.enter()
    def load(self):
        import os, torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        token = os.environ["HF_TOKEN"]
        cache = MOUNT_PATH

        print("Loading base model…")
        base = AutoModelForCausalLM.from_pretrained(
            HF_BASE,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache,
            token=token,
        )
        print("Loading Penumbra adapter…")
        self.model = PeftModel.from_pretrained(
            base, HF_ADAPTER,
            cache_dir=cache,
            token=token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_ADAPTER, cache_dir=cache, token=token
        )
        self.model.eval()
        print("✅ Penumbra ready.")

    @modal.method()
    def predict(self, question: str) -> dict:
        import json, torch

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

    @modal.method()
    def base_response(self, question: str) -> str:
        import os
        from mistralai import Mistral
        c = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        r = c.chat.complete(
            model="ministral-8b-latest",
            messages=[{"role": "user", "content": question}],
        )
        return r.choices[0].message.content


# ─── FastAPI web endpoint ─────────────────────────────────────────────────────
@app.function(secrets=secrets)
@modal.web_endpoint(method="POST", label="predict")
def predict_endpoint(body: dict) -> dict:
    """
    POST body: {"question": "your question here"}
    Returns:   uncertainty map dict, or {"base": "..."} if ?mode=base
    """
    question = body.get("question", "").strip()
    if not question:
        return {"error": "Missing 'question' field"}

    mode = body.get("mode", "penumbra")   # "penumbra" | "base"
    model = PenumbraModel()

    if mode == "base":
        return {"response": model.base_response.remote(question)}
    return model.predict.remote(question)