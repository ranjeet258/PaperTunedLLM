import os
import sys

# Adding the root path so we can import rag module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rag.document_processor import DocumentProcessor
from rag.qdrant_client import QdrantManager

class RAGService:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        host = qdrant_url.split("//")[-1].split(":")[0]
        port = int(qdrant_url.split(":")[-1])
        
        self.qdrant = QdrantManager(host=host, port=port)
        self.processor = DocumentProcessor()
        
    def process_and_store_document(self, document_id: str, title: str, file_path: str):
        """Processes a PDF and stores its chunks in Qdrant."""
        chunks_with_emb = self.processor.process_document(file_path)
        self.qdrant.insert_chunks(document_id, title, chunks_with_emb)
        return len(chunks_with_emb)
        
    def retrieve_chunks(self, query: str, limit: int = 2):
        """Retrieves relevant chunks for a given query."""
        query_embedding = self.processor.get_embeddings([query])[0]
        results = self.qdrant.search(query_embedding, limit=limit)
        
        retrieved_texts = []
        for res in results:
            retrieved_texts.append({
                "score": res.score,
                "text": res.payload["text"],
                "title": res.payload["title"],
                "document_id": res.payload["document_id"]
            })
        return retrieved_texts
        
    def build_prompt(self, query: str, retrieved_chunks: list) -> str:
        """Constructs the prompt with context for the LLM."""
        context = "\n\n".join([f"Source ({chunk['title']}):\n{chunk['text']}" for chunk in retrieved_chunks])
        
        # Format specifically for Qwen 2.5
        prompt = (
            "<|im_start|>system\n"
            "You are PaperTunedLLM, an expert AI research assistant. "
            "Use the provided context from machine learning research papers to answer the user's question. "
            "CRITICAL INSTRUCTIONS: "
            "1. You MUST answer the question in natural English. "
            "2. Keep your answer VERY concise (maximum 5 to 6 lines). "
            "3. DO NOT output dataset tags like 'FilterWhere', 'Extractive', or PDF filenames. Just output the answer directly. "
            "If the answer is not in the context, state that you don't know based on the provided papers.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Context:\n{context}\n\nQuestion:\n{query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt
