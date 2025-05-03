import json
import os
import numpy as np
from typing import Dict, List, Set, Union
from datetime import datetime

def extract_id(text: str) -> str:
    """Extract ID from text in format 'Entity: Name (ID: mid)'."""
    if "(ID: " not in text:
        return None
    return text.split("(ID: ")[1].split(")")[0].strip()

def calculate_metrics(item: Dict, k_values: List[int], debug: bool = False) -> Dict[int, dict]:
    """
    Calculate all metrics for a single question across k values
    Returns: {k: {"hits": 0/1, "precision": float, "recall": float, "f1": float}}
    """
    metrics = {k: {"hits": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0} 
               for k in k_values}
    
    try:
        if debug:
            print(f"\n{'='*50}\nProcessing question: {item.get('question', '')[:100]}...")

        # Extract correct answer IDs
        correct_ids = set()
        for ans in item.get("answers", []):
            if isinstance(ans, dict):
                correct_ids.add(ans.get("AnswerArgument", ""))
            elif isinstance(ans, str):
                ans_id = extract_id(ans)
                if ans_id:
                    correct_ids.add(ans_id)
        
        if not correct_ids:
            if debug:
                print("Warning: No valid answer IDs found")
            return metrics

        if debug:
            print(f"Ground truth IDs: {sorted(correct_ids)}")

        # Extract retrieved document IDs
        retrieved_ids = []
        for doc in item.get("rerank", []):
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            doc_id = extract_id(text)
            if doc_id:
                retrieved_ids.append(doc_id)

        if debug:
            print(f"All retrieved IDs: {retrieved_ids}")

        # Calculate metrics for each k
        for k in k_values:
            top_k = retrieved_ids[:k]
            correct_found = len(correct_ids & set(top_k))
            total_correct = len(correct_ids)
            
            # Hits@k (binary)
            metrics[k]["hits"] = 1 if correct_found > 0 else 0
            
            # Precision@k
            metrics[k]["precision"] = correct_found / k if k > 0 else 0.0
            
            # Recall@k
            metrics[k]["recall"] = correct_found / total_correct if total_correct > 0 else 0.0
            
            # F1@k
            p = metrics[k]["precision"]
            r = metrics[k]["recall"]
            metrics[k]["f1"] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

            if debug:
                print(f"\nFor k={k}:")
                print(f"Top-{k} IDs: {top_k}")
                print(f"Correct found: {correct_found}/{total_correct}")
                print(f"Precision: {metrics[k]['precision']:.2f}")
                print(f"Recall: {metrics[k]['recall']:.2f}")
                print(f"F1: {metrics[k]['f1']:.2f}")

    except Exception as e:
        print(f"Error processing item: {str(e)}")
    
    return metrics

def calculate_dataset_metrics(
    results_file: str, 
    k_values: List[int] = [2, 4, 8],
    sample_debug: int = 3
) -> Dict[int, dict]:
    """
    Calculate all metrics across dataset
    Returns: {k: {"hits": float, "precision": float, "recall": float, "f1": float}}
    """
    metrics = {k: {"hits": 0.0, "precision": 0.0, 
                   "recall": 0.0, "f1": 0.0} 
               for k in k_values}
    total_questions = 0
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                total_questions += 1
                debug = i < sample_debug
                question_metrics = calculate_metrics(item, k_values, debug=debug)
                
                # Accumulate metrics
                for k in k_values:
                    for metric in ["hits", "precision", "recall", "f1"]:
                        metrics[k][metric] += question_metrics[k][metric]
                        
            except Exception as e:
                print(f"Skipping malformed line {i+1}: {str(e)}")
                continue
    
    # Average metrics
    for k in k_values:
        for metric in metrics[k]:
            metrics[k][metric] = round(metrics[k][metric] / total_questions, 4)
    
    return metrics

def print_full_results(metrics: Dict[int, dict]):
    """Print comprehensive results in readable format"""
    print("\n{'k': {'Hits@k': 'Precision@k': 'Recall@k': 'F1@k'}}")
    for k in sorted(metrics.keys()):
        print(f"k={k}:")
        print(f"  Hits@k:    {metrics[k]['hits'] * 100:.2f}%")
        print(f"  Precision: {metrics[k]['precision'] * 100:.2f}%")
        print(f"  Recall:    {metrics[k]['recall'] * 100:.2f}%")
        print(f"  F1:        {metrics[k]['f1'] * 100:.2f}%")
        print("-" * 40)

def save_full_results(metrics: Dict[int, dict], output_file: str):
    """Save complete metrics to JSON file"""
    # Convert float32 to float for JSON serialization
    converted_metrics = {}
    for k in metrics:
        converted_metrics[k] = {metric: float(value) 
                               for metric, value in metrics[k].items()}
    
    with open(output_file, 'w') as f:
        json.dump(converted_metrics, f, indent=2)
    print(f"Full metrics saved to {output_file}")

def main():
    # Configure paths
    results_file = r"C:\Users\navad\OneDrive\Desktop\vs code\Open-Domain-QA\LLMQA\output_result\webq\test\3-reranking_evaluation.jsonl"
    output_json = r"C:\Users\navad\OneDrive\Desktop\vs code\Open-Domain-QA\LLMQA\output_result\full_metrics.json"
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return
    
    print(f"Evaluating metrics for: {results_file}")
    
    # Calculate all metrics
    metrics = calculate_dataset_metrics(
        results_file,
        k_values=[2, 4, 8],
        sample_debug=3  # Show debug for first 3 questions
    )
    
    # Print and save results
    print_full_results(metrics)
    save_full_results(metrics, output_json)

if __name__ == "__main__":
    main()