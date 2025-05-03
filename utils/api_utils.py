import os
import threading
import time
from queue import Queue
from groq import Groq
from sentence_transformers import SentenceTransformer

# Initialize Groq client
# groq_client = Groq(api_key="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")
groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")
# groq_client = Groq(api_key="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")

# Initialize local embedding model (runs locally, no API needed)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def rmreturn(s):
    s = s.replace('\n\n', ' ')
    s = s.replace('\n', ' ')
    return s.strip()

def prompt_format(prompt_template, prompt_content):
    return prompt_template.format(**prompt_content)

def api_single_request(message, model="deepseek-r1-distill-llama-70b", max_tokens=128, temperature=0.6, candidate_n=1,
                       rank=-1, result_queue=None, top_p=0.95):
    request_cnt = 0
    while True:
        request_cnt += 1
        if request_cnt > 10:  # Reduced retries
            return ""  # Return empty string instead of exiting
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False,  # Disable streaming for batch processing
                stop=None
            )
            return_response = rmreturn(response.choices[0].message.content)
            if rank == -1:
                return return_response
            else:
                result_queue.put({
                    'rank': rank,
                    'response': return_response
                })
                return
        except Exception as e:
            print(f"API Error: {str(e)}")
            if "rate limit" in str(e).lower():
                time.sleep(5)
            else:
                time.sleep(1)

def api_multi_request(prompts, model="deepseek-r1-distill-llama-70b", max_token=128, temperature=0.6, candidate_n=1,top_p=0.95):
    threads = []
    result_queue = Queue()
    gathered_response = [[] for _ in range(len(prompts))]

    for prompt_idx in range(len(prompts)):
        for _ in range(candidate_n):
            message = [{'role': 'user', 'content': prompts[prompt_idx]}]
            t = threading.Thread(
                target=api_single_request,
                args=(message, model, max_token, temperature, 1, prompt_idx, result_queue),
                kwargs={'top_p': top_p}
            )
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    while not result_queue.empty():
        item = result_queue.get()
        prompt_idx = item['rank']
        gathered_response[prompt_idx].append(item['response'])

    return gathered_response

def embedding_single_request(text):
    return embedding_model.encode(text, convert_to_tensor=False).tolist()

def embedding_multi_request(texts):
    return embedding_model.encode(texts, convert_to_tensor=False).tolist()