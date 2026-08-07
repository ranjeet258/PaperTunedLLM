from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./models/Qwen2.5-3B-QASPER-AWQ-4bit"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model to GPU (this might take a moment)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    dtype=torch.float16
)

print("\nModel loaded successfully! Ready for inference.")

prompt = "Explain what machine learning is in one short paragraph."
print(f"\nUser Prompt: {prompt}")

messages = [
    {"role": "system", "content": "You are PaperTunedLLM, a helpful AI assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating response...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256
)

# Extract only the newly generated tokens (ignore the prompt)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n--- Model Output ---")
print(response)
print("--------------------")
