import re
from rank_bm25 import BM25Okapi
from datasets import load_dataset
from transformers import pipeline

# -------------------------------
# Text Preprocessor
# -------------------------------
class TextProcessor:
    def __init__(self):
        self.tokenizer_regex = re.compile(r"\w+")

    def fast_tokenize(self, text: str):
        return self.tokenizer_regex.findall(text.lower())


# -------------------------------
# Retriever (BM25 + Chunking)
# -------------------------------
class EfficientRetriever:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.docs = []
        self.titles = []
        self.bm25 = None

    def build_corpus_fast(self, sample_size: float = 0.01, chunk_size: int = 150):
        """
        Load a subset of Wikipedia, chunk docs into passages, and index with BM25.
        sample_size=0.01 -> 1% of dataset
        chunk_size=150   -> ~150 words per passage
        """
        print("Loading and indexing corpus (fast mode)...")
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en",
                               split=f"train[:{int(sample_size * 100)}%]")

        self.docs, self.titles = [], []

        print("Processing documents with chunking...")
        for i, row in enumerate(dataset):
            text = row["text"]
            title = row["title"]

            words = text.split()
            for start in range(0, len(words), chunk_size):
                chunk = " ".join(words[start:start+chunk_size])
                if len(chunk) > 50:  # ignore too-small chunks
                    self.docs.append(chunk)
                    self.titles.append(title)

            if i % 1000 == 0 and i > 0:
                print(f"Processed {i} documents...")

        print("Building BM25 index...")
        tokenized_corpus = [self.text_processor.fast_tokenize(doc) for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ Indexed {len(self.docs)} passages")

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve top-k passages for the query.
        """
        tokenized_query = self.text_processor.fast_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [{"title": self.titles[i], "text": self.docs[i], "score": scores[i]} for i in ranked_indices]
        return results


# -------------------------------
# Reader (BERT QA model)
# -------------------------------
class BertReader:
    def __init__(self, model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
        print(f"Loading BERT QA model: {model_name}...")
        self.qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

    def get_answer(self, question: str, context: str):
        return self.qa_pipeline({"question": question, "context": context})


# -------------------------------
# ODQA System
# -------------------------------
def answer_question_odqa(question: str, retriever: EfficientRetriever, reader: BertReader, top_k: int = 5, threshold: float = 0.3):
    """
    Retrieve top-k passages, run QA reader on each, and pick best-scoring answer.
    If the best score < threshold, return "I don't know".
    """
    retrieved_docs = retriever.retrieve(question, top_k=top_k)

    answers = []
    for doc in retrieved_docs:
        try:
            result = reader.get_answer(question, doc["text"])
            answers.append({
                "answer": result["answer"],
                "score": result["score"],
                "context": doc["text"],
                "title": doc["title"]
            })
        except Exception as e:
            print(f"Reader failed on a doc: {e}")

    if not answers:
        return {"answer": "I don't know", "score": 0.0, "context": "", "title": ""}

    # Pick the best-scoring answer
    best_answer = max(answers, key=lambda x: x["score"])

    # Fallback if confidence is too low
    if best_answer["score"] < threshold:
        return {"answer": "I don't know", "score": best_answer["score"], "context": "", "title": ""}

    return best_answer



# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting traditional ODQA system...")

    # 1. Build retriever
    retriever = EfficientRetriever()
    retriever.build_corpus_fast(sample_size=0.01)  # small subset for testing

    # 2. Load reader
    reader = BertReader()

    # 3. Example query
    query = "What is the speed of dark matter?"
    result = answer_question_odqa(query, retriever, reader, top_k=5)

    print("\n🔎 Question:", query)
    print("✅ Answer:", result["answer"])
    print("📖 From:", result["title"])
    print("📝 Context snippet:", result["context"][:200], "...")
    

