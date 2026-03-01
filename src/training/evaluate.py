"""
evaluate.py — Evaluate uncertainty map quality.

Three key metrics:
1. Calibration — when model says 0.9 confidence, is it right 90% of time?
2. Human agreement — do humans agree with uncertainty assessments?
3. Before/after comparison — base vs finetuned on same questions
"""

import json
import os
import torch
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models/penumbra-ministral-8b")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_finetuned_model():
    print("Loading finetuned model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Ministral-8B-Instruct-2410",
        dtype =torch.bfloat16,
        device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base_model, str(MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    return model, tokenizer


def generate_uncertainty_map(
    model, tokenizer, question: str, max_new_tokens: int = 1024
) -> dict:
    prompt = f"<s>[INST] {question} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    try:
        return json.loads(generated)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw": generated}


def generate_base_response(question: str) -> str:
    response = client.chat.complete(
        model="ministral-8b-latest",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content


def calibration_analysis(test_data: list, model, tokenizer) -> dict:
    print("Running calibration analysis...")
    
    confidence_buckets = {
        "0.0-0.2": {"total": 0, "correct": 0},
        "0.2-0.4": {"total": 0, "correct": 0},
        "0.4-0.6": {"total": 0, "correct": 0},
        "0.6-0.8": {"total": 0, "correct": 0},
        "0.8-1.0": {"total": 0, "correct": 0},
    }
    
    for item in tqdm(test_data[:100]):
        question = item["messages"][1]["content"]
        ground_truth = json.loads(item["messages"][2]["content"])
        
        predicted = generate_uncertainty_map(model, tokenizer, question)
        
        if "error" in predicted or "claims" not in predicted:
            continue
        
        for claim in predicted["claims"]:
            conf = claim["confidence"]
            
            if conf < 0.2:
                bucket = "0.0-0.2"
            elif conf < 0.4:
                bucket = "0.2-0.4"
            elif conf < 0.6:
                bucket = "0.4-0.6"
            elif conf < 0.8:
                bucket = "0.6-0.8"
            else:
                bucket = "0.8-1.0"
            
            gt_claims = {
                c["claim"]: c["confidence"]
                for c in ground_truth.get("claims", [])
            }
            
            if claim["claim"] in gt_claims:
                gt_conf = gt_claims[claim["claim"]]
                is_correct = abs(conf - gt_conf) < 0.2
                confidence_buckets[bucket]["total"] += 1
                if is_correct:
                    confidence_buckets[bucket]["correct"] += 1
    
    calibration = {}
    for bucket, counts in confidence_buckets.items():
        if counts["total"] > 0:
            accuracy = counts["correct"] / counts["total"]
            calibration[bucket] = {
                "accuracy": accuracy,
                "total": counts["total"]
            }
    
    return calibration


def before_after_comparison(questions: list, model, tokenizer) -> list:
    print("Generating before/after comparisons...")
    
    comparisons = []
    for question in tqdm(questions):
        base_response = generate_base_response(question)
        finetuned_response = generate_uncertainty_map(model, tokenizer, question)
        
        comparisons.append({
            "question": question,
            "base_model": base_response,
            "finetuned_model": finetuned_response
        })
    
    return comparisons


def run_evaluation():
    print("=" * 50)
    print("EVALUATING UNCERTAINTY MAPS")
    print("=" * 50)
    
    test_data = []
    with open(PROCESSED_DIR / "test.jsonl") as f:
        for line in f:
            test_data.append(json.loads(line))
    test_data = test_data[:10]
    print(f"Test examples: {len(test_data)}")
    
    model, tokenizer = load_finetuned_model()
    
    calibration = calibration_analysis(test_data, model, tokenizer)
    
    with open(RESULTS_DIR / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    
    print("\nCalibration Results:")
    for bucket, data in calibration.items():
        print(f"  {bucket}: {data['accuracy']:.2f} accuracy ({data['total']} claims)")
    
    demo_questions = [
        "What caused the 2008 financial crisis?",
        "Is coffee good or bad for your health?",
        "What will AI look like in 2030?",
        "Did the Roman Empire fall due to barbarian invasions?",
        "Is nuclear energy safe?",
        "What is the speed of light?",
        "Who was the better president, Lincoln or Washington?",
        "What causes depression?",
    ]
    
    comparisons = before_after_comparison(demo_questions, model, tokenizer)
    
    with open(RESULTS_DIR / "before_after.json", "w") as f:
        json.dump(comparisons, f, indent=2)
    
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "penumbra"),
        name="evaluation"
    )
    
    cal_data = [
        [bucket, data["accuracy"], data["total"]]
        for bucket, data in calibration.items()
    ]
    wandb.log({
        "calibration_table": wandb.Table(
            columns=["confidence_bucket", "accuracy", "n_claims"],
            data=cal_data
        )
    })
    
    wandb.finish()
    print(f"\nEvaluation complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    run_evaluation()
