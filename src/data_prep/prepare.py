"""
prepare.py — Format all data into Mistral instruction tuning format.

Takes annotated + synthetic data and produces clean train/val/test splits
ready for QLoRA finetuning on Ministral 3B.
"""

import json
import random
from pathlib import Path
from tqdm import tqdm

SYNTHETIC_DIR = Path("data/synthetic")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ─────────────────────────────────────────────
# Mistral instruction format
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an epistemically transparent AI assistant.

When answering questions, you ALWAYS provide:
1. A complete, natural language answer
2. A structured uncertainty map breaking down your confidence in each claim

Your uncertainty maps reveal exactly where you are certain vs guessing.
You never hide uncertainty behind confident-sounding prose."""

def format_as_instruction(question: str, uncertainty_map: dict) -> dict:
    """
    Format a QA pair into Mistral instruction tuning format.
    
    Input:  question text
    Output: complete answer + uncertainty map JSON
    """
    user_message = f"[INST] {question} [/INST]"
    
    # Clean and validate uncertainty map
    output = {
        "answer": uncertainty_map.get("answer", ""),
        "claims": uncertainty_map.get("claims", []),
        "overall_confidence": uncertainty_map.get("overall_confidence", 0.5),
        "least_certain_claim": uncertainty_map.get("least_certain_claim", ""),
        "epistemic_summary": uncertainty_map.get("epistemic_summary", "")
    }
    
    # Validate claims have required fields
    clean_claims = []
    for claim in output["claims"]:
        if all(k in claim for k in ["claim", "confidence", "basis"]):
            clean_claims.append({
                "claim": claim["claim"],
                "confidence": float(claim["confidence"]),
                "basis": claim["basis"],
                "evidence_quality": claim.get("evidence_quality", "moderate"),
                "alternative_views": claim.get("alternative_views", None)
            })
    output["claims"] = clean_claims
    
    if not clean_claims:
        return None
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": json.dumps(output, indent=2)}
        ]
    }


def load_all_data() -> list:
    """Load all annotated and synthetic data."""
    all_data = []
    
    # Load annotated real data
    annotated_path = SYNTHETIC_DIR / "all_annotated.jsonl"
    if annotated_path.exists():
        with open(annotated_path) as f:
            for line in f:
                item = json.loads(line)
                all_data.append({
                    "question": item["question"],
                    "uncertainty_map": item["uncertainty_map"],
                    "source": item["source"]
                })
        print(f"Loaded {len(all_data)} annotated examples")
    
    # Load synthetic data
    synthetic_path = SYNTHETIC_DIR / "synthetic_generated.jsonl"
    if synthetic_path.exists():
        synthetic_count = 0
        with open(synthetic_path) as f:
            for line in f:
                item = json.loads(line)
                all_data.append({
                    "question": item["question"],
                    "uncertainty_map": item["uncertainty_map"],
                    "source": "synthetic"
                })
                synthetic_count += 1
        print(f"Loaded {synthetic_count} synthetic examples")
    
    return all_data


def prepare_all():
    print("=" * 50)
    print("PREPARING TRAINING DATA")
    print("=" * 50)
    
    # Load all data
    raw_data = load_all_data()
    print(f"\nTotal raw examples: {len(raw_data)}")
    
    # Format as instruction tuning examples
    formatted = []
    skipped = 0
    
    for item in tqdm(raw_data):
        example = format_as_instruction(
            item["question"],
            item["uncertainty_map"]
        )
        if example:
            formatted.append(example)
        else:
            skipped += 1
    
    print(f"Formatted: {len(formatted)} | Skipped: {skipped}")
    
    # Shuffle
    random.shuffle(formatted)
    
    # Split 80/10/10
    n = len(formatted)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    splits = {
        "train": formatted[:train_end],
        "val": formatted[train_end:val_end],
        "test": formatted[val_end:]
    }
    
    # Save splits
    for split_name, split_data in splits.items():
        path = PROCESSED_DIR / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        print(f"  {split_name}: {len(split_data)} examples → {path}")
    
    # Save stats
    stats = {
        "total": len(formatted),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "skipped": skipped
    }
    
    with open(PROCESSED_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Data preparation complete")
    print(f"   Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")
    
    return splits


if __name__ == "__main__":
    prepare_all()
