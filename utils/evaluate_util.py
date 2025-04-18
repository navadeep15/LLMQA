# import numpy as np


# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis=0)


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


# def cosine_similarity(x1, x2):
#     return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# def calculate_single_recall(item, k):
#     # Extract correct answer IDs
#     correct_ids = {
#         ans.split("(ID: ")[1].split(")")[0] 
#         for ans in item["answers"]
#     }
    
#     # Extract retrieved document IDs
#     retrieved_ids = [
#         doc["text"].split("(ID: ")[1].split(")")[0]
#         for doc in item["rerank"]
#     ]
    
#     # Check matches in top-k
#     top_k = retrieved_ids[:k]
#     return 100.0 if len(correct_ids & set(top_k)) > 0 else 0.0

# # Usage for your example:
# recall_2 = calculate_single_recall(item, 2)  # 100.0
# recall_4 = calculate_single_recall(item, 4)  # 100.0
# recall_8 = calculate_single_recall(item, 8)  # 100.0


import numpy as np
import json
from typing import Dict, List, Union

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

def calculate_single_recall(item: Dict, k: int) -> float:
    """
    Calculate recall@k for a single question.
    
    Args:
        item: Dictionary containing 'answers' and 'rerank' fields
        k: Number of top documents to consider
        
    Returns:
        Recall percentage (0-100)
    """
    try:
        # Extract correct answer IDs (handle both string and dict formats)
        correct_ids = set()
        for ans in item["answers"]:
            if isinstance(ans, dict):
                correct_ids.add(ans["AnswerArgument"])  # WebQSP format
            else:
                # Handle "Entity: Name (ID: mid)" format
                if "(ID: " in ans:
                    correct_ids.add(ans.split("(ID: ")[1].split(")")[0])
        
        # Extract retrieved document IDs
        retrieved_ids = []
        for doc in item["rerank"]:
            text = doc["text"] if isinstance(doc, dict) else doc
            if "(ID: " in text:
                retrieved_ids.append(text.split("(ID: ")[1].split(")")[0])
        
        # Calculate recall
        top_k = retrieved_ids[:k]
        return 100.0 if len(correct_ids & set(top_k)) > 0 else 0.0
    
    except Exception as e:
        print(f"Error calculating recall: {e}")
        return 0.0

def calculate_dataset_recall(results_file: str, k_values: List[int] = [2, 4, 8]) -> Dict[int, float]:
    """
    Calculate recall metrics across entire dataset.
    
    Args:
        results_file: Path to 3-reranking_evaluation.jsonl
        k_values: List of k values to calculate recall for
        
    Returns:
        Dictionary of {k: recall_percentage}
    """
    recall_counts = {k: 0 for k in k_values}
    total_questions = 0
    
    with open(results_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                total_questions += 1
                
                for k in k_values:
                    if calculate_single_recall(item, k) > 0:
                        recall_counts[k] += 1
                        
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line}")
                continue
    
    return {k: round((count/total_questions)*100, 2) for k, count in recall_counts.items()}