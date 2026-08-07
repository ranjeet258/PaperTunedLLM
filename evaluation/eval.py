import requests
import time

def evaluate_model():
    """
    Evaluates the model by sending a set of predefined QA pairs to the vLLM endpoint.
    """
    # Assuming vLLM is running locally via Docker Compose
    VLLM_API_URL = "http://localhost:8000/v1/completions"
    
    test_prompts = [
        "Explain the core concept of LoRA (Low-Rank Adaptation).",
        "What is the difference between AWQ and GPTQ quantization?",
        "How does Retrieval-Augmented Generation improve LLM responses?"
    ]
    
    print("Starting Model Evaluation...")
    for prompt in test_prompts:
        print(f"\n[Prompt]: {prompt}")
        
        payload = {
            "model": "/models/PaperTunedLLM-AWQ",
            "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "max_tokens": 200,
            "temperature": 0.2
        }
        
        start_time = time.time()
        try:
            response = requests.post(VLLM_API_URL, json=payload)
            response.raise_for_status()
            
            result = response.json()
            completion = result['choices'][0]['text']
            
            print(f"[Response]: {completion.strip()}")
            print(f"[Latency]: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    evaluate_model()
