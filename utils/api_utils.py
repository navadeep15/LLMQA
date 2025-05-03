import os
import threading
import time
from queue import Queue
from groq import Groq
from sentence_transformers import SentenceTransformer

# Initialize Groq client
groq_client = Groq(api_key="")  ## put your api key here

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