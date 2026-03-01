"""
collect.py — Pull public datasets for uncertainty map training
Sources: TruthfulQA, TriviaQA, NaturalQuestions, FEVER
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def collect_truthfulqa(max_samples=500):
    print("Collecting TruthfulQA...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")

    samples = []
    for item in tqdm(dataset):
        samples.append({
            "source": "truthfulqa",
            "question": item["question"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"],
            "category": item["category"],
            "expected_uncertainty": "high"
        })

    path = RAW_DIR / "truthfulqa.jsonl"
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"  Saved {len(samples)} TruthfulQA examples → {path}")
    return samples


def collect_triviaqa(max_samples=1000):
    print("Collecting TriviaQA...")
    dataset = load_dataset(
        "trivia_qa", "rc.nocontext",
        split=f"train[:{max_samples}]"
    )

    samples = []
    for item in tqdm(dataset):
        samples.append({
            "source": "triviaqa",
            "question": item["question"],
            "correct_answers": item["answer"]["aliases"],
            "primary_answer": item["answer"]["value"],
            "expected_uncertainty": "low"
        })

    path = RAW_DIR / "triviaqa.jsonl"
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"  Saved {len(samples)} TriviaQA examples → {path}")
    return samples

def collect_fever(max_samples=500):
    """
    FEVER: Fact verification dataset.
    SUPPORTS / REFUTES / NOT ENOUGH INFO labels
    → directly maps to uncertainty levels
    """
    print("Collecting FEVER...")
    dataset = load_dataset("fever", "v1.0", split=f"train[:{max_samples}]", trust_remote_code=True)

    label_to_uncertainty = {
        "SUPPORTS": "low",
        "REFUTES": "low",  
        "NOT ENOUGH INFO": "high" 
    }

    samples = []
    for item in tqdm(dataset):
        samples.append({
            "source": "fever",
            "claim": item["claim"],
            "label": item["label"],
            "expected_uncertainty": label_to_uncertainty.get(
                item["label"], "medium"
            )
        })

    path = RAW_DIR / "fever.jsonl"
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"  Saved {len(samples)} FEVER examples → {path}")
    return samples


def collect_all():
    print("=" * 50)
    print("COLLECTING ALL DATASETS")
    print("=" * 50)

    results = {}
    results["truthfulqa"] = collect_truthfulqa()
    results["triviaqa"] = collect_triviaqa()
    results["fever"] = collect_fever()

    total = sum(len(v) for v in results.values())
    print(f"\n✅ Total collected: {total} examples")
    return results


if __name__ == "__main__":
    collect_all()
