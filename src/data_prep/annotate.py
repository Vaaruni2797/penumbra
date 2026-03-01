"""
annotate.py — Use Mistral Large 3 to annotate each QA pair
with structured uncertainty maps.

This is the teacher that teaches Ministral 3B.
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

RAW_DIR = Path("data/raw")
SYNTHETIC_DIR = Path("data/synthetic")
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_PROMPT = """You are an expert at epistemic analysis.

Given a question and its correct answer, decompose the answer into 
individual claims and assess the uncertainty of each claim.

Question: {question}
Answer: {answer}

For each distinct claim in the answer, assess:
1. confidence (0.0-1.0): How certain is this claim?
   - 0.9-1.0: Mathematical/logical certainty or overwhelming consensus
   - 0.7-0.9: Strong evidence, mainstream scientific consensus  
   - 0.5-0.7: Good evidence but some legitimate debate
   - 0.3-0.5: Contested, multiple reasonable views exist
   - 0.0-0.3: Highly uncertain, speculative, or poorly evidenced

2. basis: Why are you confident or uncertain? (1-2 sentences)

3. evidence_quality: "strong" | "moderate" | "weak" | "contested"

4. alternative_views: If confidence < 0.6, what do others argue?

Return ONLY valid JSON in exactly this format:
{{
  "answer": "the complete natural language answer",
  "claims": [
    {{
      "claim": "specific assertion being made",
      "confidence": 0.85,
      "basis": "why this confidence level",
      "evidence_quality": "strong",
      "alternative_views": null
    }}
  ],
  "overall_confidence": 0.78,
  "least_certain_claim": "the claim with lowest confidence",
  "epistemic_summary": "one sentence about overall certainty of this answer"
}}"""


def annotate_qa_pair(question: str, answer: str, retries: int = 3) -> dict:
    prompt = ANNOTATION_PROMPT.format(
        question=question,
        answer=answer
    )

    for attempt in range(retries):
        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"  Failed to annotate after {retries} attempts: {e}")
            return None


def annotate_truthfulqa():
    print("Annotating TruthfulQA...")
    path = RAW_DIR / "truthfulqa.jsonl"

    if not path.exists():
        print("  TruthfulQA not found. Run collect.py first.")
        return []

    with open(path) as f:
        raw = [json.loads(line) for line in f]

    annotated = []
    for item in tqdm(raw[:200]):
        if not item["correct_answers"]:
            continue

        answer = item["correct_answers"][0]
        result = annotate_qa_pair(item["question"], answer)

        if result:
            annotated.append({
                "source": "truthfulqa",
                "question": item["question"],
                "uncertainty_map": result,
                "expected_uncertainty_level": item["expected_uncertainty"]
            })

        time.sleep(0.5)
    return annotated


def annotate_triviaqa():
    print("Annotating TriviaQA...")
    path = RAW_DIR / "triviaqa.jsonl"

    if not path.exists():
        print("  TriviaQA not found. Run collect.py first.")
        return []

    with open(path) as f:
        raw = [json.loads(line) for line in f]

    annotated = []
    for item in tqdm(raw[:300]):
        result = annotate_qa_pair(
            item["question"],
            item["primary_answer"]
        )

        if result:
            annotated.append({
                "source": "triviaqa",
                "question": item["question"],
                "uncertainty_map": result,
                "expected_uncertainty_level": item["expected_uncertainty"]
            })

        time.sleep(0.5)

    return annotated

def save_annotated(annotated: list, filename: str):
    path = SYNTHETIC_DIR / filename
    with open(path, "w") as f:
        for item in annotated:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(annotated)} annotated examples → {path}")


def annotate_all():
    print("=" * 50)
    print("ANNOTATING WITH MISTRAL LARGE 3")
    print("=" * 50)

    all_annotated = []

    truthfulqa_annotated = annotate_truthfulqa()
    save_annotated(truthfulqa_annotated, "truthfulqa_annotated.jsonl")
    all_annotated.extend(truthfulqa_annotated)

    triviaqa_annotated = annotate_triviaqa()
    save_annotated(triviaqa_annotated, "triviaqa_annotated.jsonl")
    all_annotated.extend(triviaqa_annotated)

    save_annotated(all_annotated, "all_annotated.jsonl")
    print(f"\nTotal annotated: {len(all_annotated)} examples")
    return all_annotated


if __name__ == "__main__":
    annotate_all()
