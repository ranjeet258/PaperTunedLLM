import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM

def evaluate_model(model, tokenizer, prompt, model_name):
    print(f"\n--- Evaluating {model_name} ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.2, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    latency = time.time() - start_time
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"Latency: {latency:.2f} seconds")
    print(f"Response:\n{response.strip()}")
    print("-" * 40)
    
def main():
    # Model Paths
    BASE_MODEL_ID = "Qwen/Qwen2.5-3B"
    AWQ_MODEL_PATH = "ranjeet258/Qwen2.5-3B-QASPER-AWQ-4bit"
    
    test_prompts = [
        "<|im_start|>user\nWhat is the seed lexicon in the context of affective events?\n<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how discourse relations propagate affective polarity.\n<|im_end|>\n<|im_start|>assistant\n"
    ]
    
    # 1. Evaluate Base Model (Unquantized)
    print("Loading Base Model (fp16)...")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        for prompt in test_prompts:
            evaluate_model(base_model, base_tokenizer, prompt, "Base Model (Qwen2.5-3B)")
            
        # Free up memory before loading the next model
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Could not evaluate base model: {e}")

    # 2. Evaluate AWQ Fine-Tuned Model
    print("\nLoading AWQ Fine-Tuned Model (4-bit)...")
    try:
        awq_tokenizer = AutoTokenizer.from_pretrained(AWQ_MODEL_PATH)
        awq_model = AutoAWQForCausalLM.from_pretrained(
            AWQ_MODEL_PATH, 
            fuse_layers=True, # AWQ optimization
            device_map="auto"
        )
        for prompt in test_prompts:
            evaluate_model(awq_model, awq_tokenizer, prompt, "Fine-Tuned AWQ Model")
            
    except Exception as e:
        print(f"Could not evaluate AWQ model: {e}")
        print(f"Make sure you have downloaded the AWQ files to {AWQ_MODEL_PATH}")

if __name__ == "__main__":
    main()
