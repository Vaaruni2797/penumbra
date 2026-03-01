"""
generate.py — Synthetic data generation using Mistral Large 3.

Generates QA pairs spanning high/medium/low certainty domains
to ensure the finetuned model sees the full uncertainty spectrum.
"""

import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
SYNTHETIC_DIR = Path("data/synthetic")
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Domain templates for each uncertainty level
# ─────────────────────────────────────────────
DOMAIN_CONFIGS = {
    "high_confidence": {
        "domains": [
            "mathematics and formal logic",
            "well-established physics (classical mechanics, thermodynamics)",
            "historical events with clear documentation (ancient history)",
            "geography (capitals, continents, major landmarks)",
            "basic chemistry (periodic table, molecular formulas)"
        ],
        "target_confidence": "0.85-1.0",
        "count": 5
    },
    "medium_confidence": {
        "domains": [
            "recent scientific research (last 10 years)",
            "economics and market behavior",
            "psychology and behavioral science",
            "nutrition and health research",
            "recent historical events (last 50 years)"
        ],
        "target_confidence": "0.5-0.8",
        "count": 5
    },
    "low_confidence": {
        "domains": [
            "predictions about future events",
            "contested historical interpretations",
            "emerging scientific fields (dark matter, consciousness)",
            "complex social and political questions",
            "cutting-edge AI research findings"
        ],
        "target_confidence": "0.2-0.5",
        "count": 5
    }
}

GENERATION_PROMPT = """Generate a single question-answer pair about {domain}.

The answer should have approximately {target_confidence} confidence level.

Return ONLY valid JSON in exactly this format, nothing else:
{{
  "question": "the question",
  "uncertainty_map": {{
    "answer": "complete natural language answer",
    "claims": [
      {{
        "claim": "specific assertion",
        "confidence": 0.85,
        "basis": "why this confidence level",
        "evidence_quality": "strong",
        "alternative_views": null
      }}
    ],
    "overall_confidence": 0.85,
    "least_certain_claim": "lowest confidence claim text",
    "epistemic_summary": "one sentence about overall certainty"
  }}
}}"""


def generate_single(level: str, domain: str) -> dict:
    """Generate one QA pair for a domain."""
    config = DOMAIN_CONFIGS[level]

    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{
                "role": "user",
                "content": GENERATION_PROMPT.format(
                    domain=domain,
                    target_confidence=config["target_confidence"]
                )
            }],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1000
        )

        parsed = json.loads(response.choices[0].message.content)
        parsed["source"] = "synthetic"
        parsed["uncertainty_level"] = level
        parsed["domain"] = domain
        return parsed

    except Exception as e:
        print(f"  Error: {e}")
        return None


def generate_for_domain(level: str, domain: str, count: int) -> list:
    """Generate `count` examples one at a time."""
    results = []
    for _ in range(count):
        example = generate_single(level, domain)
        if example:
            results.append(example)
        time.sleep(0.3)
    return results


def generate_all():
    print("=" * 50)
    print("GENERATING SYNTHETIC DATA WITH MISTRAL LARGE 3")
    print("=" * 50)

    all_synthetic = []

    for level, config in DOMAIN_CONFIGS.items():
        print(f"\nGenerating {level} examples...")
        count_per_domain = max(3, config["count"] // len(config["domains"]))

        for domain in tqdm(config["domains"]):
            examples = generate_for_domain(level, domain, count_per_domain)
            all_synthetic.extend(examples)
            print(f"  {domain}: {len(examples)} examples")
            time.sleep(1)  # Rate limiting

    # Save
    path = SYNTHETIC_DIR / "synthetic_generated.jsonl"
    with open(path, "w") as f:
        for item in all_synthetic:
            f.write(json.dumps(item) + "\n")

    print(f"\n✅ Total synthetic: {len(all_synthetic)} examples → {path}")
    return all_synthetic


if __name__ == "__main__":
    generate_all()