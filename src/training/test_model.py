# test_model.py — run from project root
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "mistralai/Ministral-8B-Instruct-2410",
    dtype=torch.bfloat16,
    device_map={"": 0}
)
model = PeftModel.from_pretrained(base, "models/penumbra-ministral-8b")
tokenizer = AutoTokenizer.from_pretrained("models/penumbra-ministral-8b")

question = "Is nuclear energy safe?"
system = """You are an epistemically transparent AI assistant.
When answering questions, you ALWAYS respond with ONLY valid JSON in exactly this format, nothing else:
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
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))