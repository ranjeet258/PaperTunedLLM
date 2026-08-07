import os
import json
import aiohttp
from typing import AsyncGenerator

class LLMService:
    def __init__(self):
        self.vllm_url = os.getenv("VLLM_URL", "http://localhost:8000/v1")
        self.model_name = "ranjeet258/Qwen2.5-3B-QASPER-AWQ-4bit" # Default, can be fetched dynamically
        
    async def generate_response_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Streams the response from the vLLM server."""
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.3,
            "repetition_penalty": 1.15,
            "stream": True,
            "stop": ["<|im_end|>", "<|endoftext|>", "Human:", "\n\n\n"]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.vllm_url}/completions", headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield f"Error: Failed to get response from LLM. Status: {response.status}, Detail: {error_text}"
                    return
                    
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            json_data = json.loads(data)
                            token = json_data['choices'][0].get('text', '')
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
