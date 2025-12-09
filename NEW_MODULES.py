import os
import json
from typing import List, Dict

try:
    from groq import Groq
except ImportError:
    print("Please install groq: pip install groq")

# Initialize Client
# Use your actual API Key here or set it in your environment variables
client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
MODEL_ID = "llama-3.3-70b-versatile"

def get_json_response(system_prompt: str, user_content: str):
    """Helper function to get JSON output from Groq."""
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        model=MODEL_ID,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# ---------------------------------------------------------
# 1. Query Decomposition Module (Slide 13)
# ---------------------------------------------------------
def decompose_query(complex_query: str) -> List[str]:
    """Breaks a complex question into sub-questions."""
    prompt = "You are a query decomposer. Break the input complex question into 2-3 simple sub-questions. Return JSON: {'sub_queries': ['q1', 'q2']}"
    return get_json_response(prompt, complex_query).get("sub_queries", [])

# ---------------------------------------------------------
# 2. Answer Verification Module (Slide 14)
# ---------------------------------------------------------
def verify_answer(question: str, answer: str, context: str) -> Dict:
    """Checks if the answer is supported by the context."""
    prompt = (
        "Verify if the answer is supported by the context. "
        "Return JSON: {'status': 'Supported'|'Contradicted'|'Uncertain', 'reason': '...'}"
    )
    content = f"Question: {question}\nAnswer: {answer}\nContext: {context}"
    return get_json_response(prompt, content)

# ---------------------------------------------------------
# 3. Document Summarization Layer (Slide 15)
# ---------------------------------------------------------
def summarize_docs(docs: List[str]) -> str:
    """Aggregates multiple docs into a single evidence summary."""
    combined_text = "\n".join(docs)[:6000]
    prompt = "Summarize the following search results into a single coherent paragraph containing only factual evidence."
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": combined_text}
        ],
        model=MODEL_ID
    )
    return response.choices[0].message.content

# ---------------------------------------------------------
# 4. Query Reformulation / Paraphrasing (Slide 16)
# ---------------------------------------------------------
def reformulate_query(query: str) -> List[str]:
    """Generates 3 paraphrased variations of the query."""
    prompt = "Generate 3 paraphrased versions of the user query to improve search recall. Return JSON: {'variations': ['v1', 'v2', 'v3']}"
    return get_json_response(prompt, query).get("variations", [])

# ---------------------------------------------------------
# 5. Evidence Diversity Scoring (Slide 17) [NEW]
# ---------------------------------------------------------
def calculate_diversity_score(documents: List[str]) -> Dict:
    """
    Analyzes how diverse the viewpoints in the documents are.
    (Simulates embedding cosine similarity using LLM reasoning for the demo).
    """
    prompt = (
        "Analyze the semantic diversity of the provided documents. "
        "Are they repeating the same info, or do they cover different aspects? "
        "Return JSON: {'diversity_score': <float 0.0 to 1.0>, 'analysis': '...'} "
        "A score of 1.0 means highly diverse/distinct, 0.0 means identical/redundant."
    )
    content = "\n---\n".join(documents)
    return get_json_response(prompt, content)

# ---------------------------------------------------------
# 6. Answer-Type Detection (Slide 18)
# ---------------------------------------------------------
def detect_answer_type(question: str) -> str:
    """Predicts expected answer format."""
    prompt = "Classify the expected answer type. Categories: [PERSON, DATE, LOCATION, NUMBER, FACT, EXPLANATION]. Return JSON: {'type': 'CATEGORY'}"
    return get_json_response(prompt, question).get("type", "UNKNOWN")

# ---------------------------------------------------------
# 7. Error Taxonomy & Categorization (Slide 19) [NEW]
# ---------------------------------------------------------
def categorize_error(question: str, wrong_answer: str, correct_context: str) -> Dict:
    """
    Diagnoses why the system failed. 
    Categories: [Retrieval Failure, Extraction Error, Reasoning Error, Hallucination].
    """
    prompt = (
        "Analyze why the system gave the 'Wrong Answer' given the 'Context'. "
        "Classify the error. Return JSON: {'category': '...', 'explanation': '...'}"
    )
    content = f"Question: {question}\nWrong Answer: {wrong_answer}\nContext: {correct_context}"
    return get_json_response(prompt, content)


# ---------------------------------------------------------
# Test Runner (Demonstrates all 7 Modules)
# ---------------------------------------------------------
if __name__ == "__main__":

    # 1. Decomposition
    q1 = "Who is the CEO of the company that created ChatGPT and when was he born?"
    print(f"--- 1. Query Decomposition ---\nInput: {q1}")
    print(decompose_query(q1))

    # 2. Reformulation
    q2 = "How to fix a flat tire?"
    print(f"\n--- 2. Query Reformulation ---\nInput: {q2}")
    print(reformulate_query(q2))

    # 3. Type Detection
    q3 = "In what year did India gain independence?"
    print(f"\n--- 3. Answer-Type Detection ---\nInput: {q3}")
    print(detect_answer_type(q3))

    # 4. Diversity Scoring (New)
    docs_diverse = [
        "Solar energy is renewable and clean.",
        "Solar panels can be expensive to install initially.",
        "Germany is a leading producer of solar power."
    ]
    print(f"\n--- 4. Evidence Diversity Scoring ---\nDocs: {docs_diverse}")
    print(calculate_diversity_score(docs_diverse))

    # 5. Summarization
    print(f"\n--- 5. Document Summarization ---\n(Summarizing the above docs...)")
    print(summarize_docs(docs_diverse))

    # 6. Verification
    q_ver = "When was the iPhone released?"
    ctx_ver = "Apple announced the iPhone on Jan 9, 2007, and released it on June 29, 2007."
    ans_ver = "The iPhone was released in 2005." # Intentionally wrong
    print(f"\n--- 6. Answer Verification ---\nClaim: {ans_ver}")
    print(verify_answer(q_ver, ans_ver, ctx_ver))

    # 7. Error Categorization (New)
    print(f"\n--- 7. Error Taxonomy Tool ---\n(Diagnosing the error above...)")
    print(categorize_error(q_ver, ans_ver, ctx_ver))