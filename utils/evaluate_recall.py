# import json
# from evaluate_util import calculate_dataset_recall

# def main():
#     # Path to your results file
#     # results_file = "output_results/webq/test/3-reranking_evaluation.jsonl"
#     results_file = r"C:\Users\navad\OneDrive\Desktop\vs code\Open-Domain-QA\LLMQA\output_result\webq\test\3-reranking_evaluation.jsonl"
#     if not os.path.exists(results_file):
#         print(f"File not found: {results_file}")
#         return
#     # Calculate recall metrics
#     recall_metrics = calculate_dataset_recall(results_file)
    
#     # Print results in the same format as the paper
#     print("\nRecall for Evidence Quality:")
#     print(f"{'Dataset':<10} {'Method':<20} {'Top-2':<8} {'Top-4':<8} {'Top-8':<8}")
#     print(f"{'WebQ':<10} {'LLMQA':<20} {recall_metrics[2]:<8.2f} {recall_metrics[4]:<8.2f} {recall_metrics[8]:<8.2f}")

# if __name__ == "__main__":
#     main()


import json
import os
import numpy as np
from typing import Dict, List, Set, Union

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # Numerical stability improvement
    return e_x / e_x.sum(axis=0)

def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute sigmoid function."""
    return 1 / (1 + np.exp(-x))

def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def extract_id(text: str) -> str:
    """Extract ID from text in format 'Entity: Name (ID: mid)'."""
    if "(ID: " not in text:
        return None
    return text.split("(ID: ")[1].split(")")[0].strip()

def calculate_single_recall(item: Dict, k: int, debug: bool = False) -> float:
    """
    Calculate recall@k for a single question with debug output.
    
    Args:
        item: Dictionary containing question data
        k: Number of top documents to consider
        debug: Whether to print debug information
        
    Returns:
        Recall percentage (0-100)
    """
    if debug:
        print(f"\n{'='*50}\nProcessing question: {item.get('question', '')[:100]}...")
    
    try:
        # Extract correct answer IDs
        correct_ids: Set[str] = set()
        for ans in item.get("answers", []):
            if isinstance(ans, dict):
                correct_ids.add(ans.get("AnswerArgument", ""))
            elif isinstance(ans, str):
                ans_id = extract_id(ans)
                if ans_id:
                    correct_ids.add(ans_id)
        
        if debug:
            print(f"Ground truth IDs: {correct_ids}")

        # Extract retrieved document IDs
        retrieved_ids: List[str] = []
        for doc in item.get("rerank", [])[:k*2]:  # Check extra docs for debugging
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            doc_id = extract_id(text)
            if doc_id:
                retrieved_ids.append(doc_id)
        
        if debug:
            print(f"Top {k} retrieved IDs: {retrieved_ids[:k]}")
            print(f"All retrieved IDs: {retrieved_ids}")

        # Calculate recall
        top_k = retrieved_ids[:k]
        recall = 100.0 if len(correct_ids & set(top_k)) > 0 else 0.0
        
        if debug and recall == 0:
            print("!! Zero recall detected !!")
            print(f"Missing IDs: {correct_ids - set(retrieved_ids)}")
        
        return recall
    
    except Exception as e:
        if debug:
            print(f"Error calculating recall: {str(e)}")
        return 0.0

def calculate_dataset_recall(
    results_file: str, 
    k_values: List[int] = [2, 4, 8],
    sample_debug: int = 3
) -> Dict[int, float]:
    """
    Calculate recall metrics across entire dataset with debugging.
    
    Args:
        results_file: Path to JSONL results file
        k_values: List of k values to calculate
        sample_debug: Number of samples to show debug info for
        
    Returns:
        Dictionary of {k: recall_percentage}
    """
    if not os.path.exists(results_file):
        print(f"Error: File not found at {results_file}")
        return {k: 0.0 for k in k_values}
    
    recall_counts = {k: 0 for k in k_values}
    total_questions = 0
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                total_questions += 1
                
                # Show debug for first N samples
                debug = i < sample_debug
                if debug:
                    print(f"\n{'#'*20} Processing Question {i+1} {'#'*20}")
                
                for k in k_values:
                    recall = calculate_single_recall(item, k, debug=debug)
                    if recall > 0:
                        recall_counts[k] += 1
                        
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {i+1}: {str(e)}")
                continue
    
    if total_questions == 0:
        print("Error: No valid questions found in the file")
        return {k: 0.0 for k in k_values}
    
    return {k: round((count/total_questions)*100, 2) for k, count in recall_counts.items()}

def print_results(recall_metrics: Dict[int, float]):
    """Print results in paper-compatible format."""
    print("\nRecall for Evidence Quality:")
    print(f"{'Dataset':<10} {'Method':<20} {'Top-2':<8} {'Top-4':<8} {'Top-8':<8}")
    print(f"{'WebQ':<10} {'LLMQA':<20}", end="")
    for k in [2, 4, 8]:
        print(f"{recall_metrics.get(k, 0):<8.2f}", end="")
    print()

def main():
    # Configure your paths here
    results_file =  r"C:\Users\navad\OneDrive\Desktop\vs code\Open-Domain-QA\LLMQA\output_result\webq\test\3-reranking_evaluation.jsonl"
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        print("Please verify the path and try again.")
        return
    
    print(f"Evaluating recall from: {results_file}")
    recall_metrics = calculate_dataset_recall(results_file, sample_debug=3)
    print_results(recall_metrics)

if __name__ == "__main__":
    main()