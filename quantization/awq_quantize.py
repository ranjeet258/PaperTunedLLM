from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

def quantize_model(model_path: str, quant_path: str):
    """
    Quantizes a model using Activation-aware Weight Quantization (AWQ).
    """
    print(f"Loading model from {model_path} for AWQ quantization...")
    
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }
    
    # Load model and tokenizer
    model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Quantize
    print("Starting quantization...")
    model.quantize(tokenizer, quant_config=quant_config)
    
    # Save quantized model
    print(f"Saving quantized model to {quant_path}...")
    os.makedirs(quant_path, exist_ok=True)
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    
    print("Quantization complete!")

if __name__ == "__main__":
    # Ensure this is run after merging the LoRA weights into the base model
    # LlamaFactory can export the merged model.
    MERGED_MODEL_PATH = "saves/Qwen2.5-3B/lora/sft/merged"
    QUANTIZED_MODEL_PATH = "models/PaperTunedLLM-AWQ"
    
    if os.path.exists(MERGED_MODEL_PATH):
        quantize_model(MERGED_MODEL_PATH, QUANTIZED_MODEL_PATH)
    else:
        print(f"Merged model not found at {MERGED_MODEL_PATH}. Please merge LoRA weights first.")
