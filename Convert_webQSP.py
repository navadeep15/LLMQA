# webq_converter.py
import json
from collections import defaultdict

def convert_webqsp(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output = []
    for q in data["Questions"]:
        # Track unique entities using (MID, Name) pairs
        entity_store = defaultdict(set)
        answers = set()
        
        for parse in q["Parses"]:
            # Add topic entity
            topic_mid = parse["TopicEntityMid"]
            topic_name = parse["TopicEntityName"]
            entity_store[topic_mid].add(topic_name)
            
            # Add answer entities
            for ans in parse.get("Answers", []):
                ans_mid = ans["AnswerArgument"]
                ans_name = ans["EntityName"]
                entity_store[ans_mid].add(ans_name)
                answers.add(f"{ans_name} (ID: {ans_mid})")

        # Create context documents with merged names
        ctxs = []
        for mid, names in entity_store.items():
            primary_name = next(iter(names))  # Get first name
            ctxs.append({
                "text": f"Entity: {primary_name} (ID: {mid})"
            })

        output.append({
            "question": q["RawQuestion"],
            "ctxs": ctxs,
            "answers": list(answers)
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

# Usage:
convert_webqsp("WebQSP.test.json", "input_data/webq/test_full_new.json")