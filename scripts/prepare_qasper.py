import json
import os
from datasets import load_dataset

def prepare_qasper_for_llamafactory(output_path: str):
    """
    Loads the QASPER dataset and formats it for LlamaFactory SFT.
    LlamaFactory format requires a JSON/JSONL with at minimum 'instruction', 'input', and 'output'.
    """
    print("Loading QASPER dataset...")
    # Using the AllenAI QASPER dataset
    dataset = load_dataset("allenai/qasper", split="train")
    
    formatted_data = []
    
    print(f"Processing {len(dataset)} examples...")
    for item in dataset:
        title = item.get('title', '')
        abstract = item.get('abstract', '')
        
        # QASPER has multiple questions per paper
        for qa_pair in item['qas']:
            question = qa_pair['question']
            
            # Extract answers. QASPER answers can be free-form, extractive, etc.
            # For simplicity in this pipeline, we will extract the first free-form answer 
            # or extractive span available.
            answer_text = ""
            for answer in qa_pair['answers']:
                if answer['answer']['free_form_answer']:
                    answer_text = answer['answer']['free_form_answer']
                    break
                elif answer['answer']['extractive_spans']:
                    answer_text = ", ".join(answer['answer']['extractive_spans'])
                    break
                    
            if not answer_text:
                continue # Skip if no valid answer format found
                
            # Construct the LlamaFactory format
            # Instruction: The task we want the model to do
            instruction = f"Answer the following question based on the provided machine learning research paper context."
            
            # Input: The context (Title + Abstract or full text)
            input_context = f"Title: {title}\nAbstract: {abstract}"
            
            # Output: The ground truth answer
            output = answer_text
            
            formatted_data.append({
                "instruction": instruction,
                "input": input_context,
                "output": output
            })

    print(f"Generated {len(formatted_data)} instruction-following examples.")
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
    print(f"Dataset successfully saved to {output_path}")

if __name__ == "__main__":
    output_file = os.path.join("datasets", "qasper_llamafactory.json")
    prepare_qasper_for_llamafactory(output_file)
