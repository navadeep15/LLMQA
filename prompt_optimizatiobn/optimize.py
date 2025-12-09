#!/usr/bin/env python3
"""
optimize_prompts.py

Usage:
    export GROQ_API_KEY="sk-..."
    python 3.py --inputs generate_expansion_prompt.jsonl document_reranking_prompt.jsonl
"""

import os
import json
import time
import logging
import argparse
from typing import Dict, Any, List

try:
    from groq import Groq
except Exception:
    raise ImportError("Install Groq SDK first: pip install groq")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------
# Evaluation + Rewrite instructions
# -------------------------
EVALUATOR_SYSTEM = (
    "You are an expert prompt evaluator. Given a JSON object with keys 'template' and 'standard', "
    "score it on this rubric (0.0-1.0): clarity, specificity, usefulness, format_validity. "
    "Also provide 'overall' (float) and 'notes' (list of short suggestions). "
    "Respond ONLY with valid JSON."
)

REWRITE_SYSTEM = (
    "You are an expert prompt engineer. Rewrite the given JSON prompt so its evaluation improves. "
    "Preserve placeholders like {question}, keep only keys 'template' and 'standard'. "
    "Respond ONLY with valid JSON."
)

# -------------------------
# Helpers
# -------------------------
def make_client(api_key: str = None) -> Groq:
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Pass --api_key or set GROQ_API_KEY env var.")
    return Groq(api_key=api_key)

def groq_json_call(client: Groq, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},  # JSON mode ensures strict output
    )
    return json.loads(resp.choices[0].message.content)

def evaluate_prompt(client: Groq, prompt: Dict[str, Any], model: str) -> Dict[str, Any]:
    return groq_json_call(client, [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
    ], model)

def rewrite_prompt(client: Groq, prompt: Dict[str, Any], evaluation: Dict[str, Any], model: str) -> Dict[str, Any]:
    return groq_json_call(client, [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": json.dumps({"prompt": prompt, "evaluation": evaluation}, ensure_ascii=False)}
    ], model)

def optimize_prompt(client: Groq, prompt: Dict[str, Any], model: str, max_iter: int = 3) -> Dict[str, Any]:
    best = prompt
    best_eval = evaluate_prompt(client, best, model)
    logging.info("  Initial score: %.3f", best_eval.get("overall", 0.0))

    for i in range(max_iter):
        new = rewrite_prompt(client, best, best_eval, model)
        new_eval = evaluate_prompt(client, new, model)
        if new_eval.get("overall", 0.0) > best_eval.get("overall", 0.0):
            logging.info("  Iteration %d improved: %.3f → %.3f", i+1, best_eval["overall"], new_eval["overall"])
            best, best_eval = new, new_eval
        else:
            logging.info("  Iteration %d did not improve (%.3f)", i+1, new_eval["overall"])
        time.sleep(0.2)

    return best

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def write_jsonl(records: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

# -------------------------
# CLI / main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Optimize prompts using Groq JSON mode.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL prompt files")
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="Groq model ID")
    parser.add_argument("--api_key", default="", help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--max_iter", type=int, default=3, help="Max rewrite iterations")
    args = parser.parse_args()

    client = make_client(args.api_key)

    for path in args.inputs:
        logging.info("Optimizing file: %s", path)
        prompts = read_jsonl(path)
        optimized = [optimize_prompt(client, p, args.model, args.max_iter) for p in prompts]

        out_path = path.replace(".jsonl", ".optimized.jsonl")
        write_jsonl(optimized, out_path)
        logging.info("Wrote %d optimized prompts → %s", len(optimized), out_path)

if __name__ == "__main__":
    main()
