import json
from groq import Groq

# Initialize Groq client with your API key
client = Groq(api_key="")

# Load templates from input.jsonl
templates = []
with open("input.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        templates.append(json.loads(line.strip()))

question = "What is the difference between linear and logistic regression?"

def calculate_score(text):
    word_count = len(text.split())
    if word_count <= 30:
        return 100
    else:
        return max(0, 100 - (word_count - 30) * 2)

def optimize_templates(templates, question):
    scored_templates = []
    for t in templates:
        # Prepare the prompt
        prompt = t['template'].replace("{standard}", t['standard']).replace("{question}", question)
        
        # Generate output using the latest Groq SDK
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-120b"
        )
        
        output_text = response.choices[0].message.content.strip()
        score = calculate_score(output_text)
        scored_templates.append({
            "prompt": prompt,
            "expansion": output_text,
            "score": score
        })
    
    # Sort by score descending
    scored_templates.sort(key=lambda x: x["score"], reverse=True)
    return scored_templates

# Run optimization
results = optimize_templates(templates, question)

# Display outputs
for i, res in enumerate(results, 1):
    print(f"=== Template #{i} ===")
    print("Prompt:\n", res["prompt"])
    print("\nGenerated Expansion:\n", res["expansion"])
    print("\nOptimization Score:", res["score"])
    print("\n------------------------\n")