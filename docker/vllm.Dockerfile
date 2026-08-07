FROM vllm/vllm-openai:v0.5.4

# Qwen2.5 requires a newer version of transformers/tokenizers than what is bundled in vLLM v0.5.4
RUN pip install --upgrade transformers tokenizers
