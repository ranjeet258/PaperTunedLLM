import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

from services.llm_service import LLMService
from services.rag_service import RAGService

router = APIRouter()
llm_service = LLMService()
rag_service = RAGService()

# In-memory store for documents for demonstration (ideally use a DB like PostgreSQL)
indexed_documents = {}

class ChatRequest(BaseModel):
    query: str

class RetrieveRequest(BaseModel):
    query: str
    limit: int = 5

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main RAG endpoint with streaming response."""
    # 1. Retrieve relevant chunks
    chunks = rag_service.retrieve_chunks(request.query)
    
    # 2. Build prompt
    prompt = rag_service.build_prompt(request.query, chunks)
    
    # 3. Stream response
    return StreamingResponse(llm_service.generate_response_stream(prompt), media_type="text/event-stream")

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a paper, extracts text, chunks, embeds, and stores in Qdrant."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    document_id = str(uuid.uuid4())
    os.makedirs("../datasets/uploads", exist_ok=True)
    file_path = f"../datasets/uploads/{document_id}.pdf"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        num_chunks = rag_service.process_and_store_document(document_id, file.filename, file_path)
        indexed_documents[document_id] = {"title": file.filename, "chunks": num_chunks}
        return {"status": "success", "document_id": document_id, "chunks_indexed": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve")
async def retrieve_chunks(request: RetrieveRequest):
    """Returns retrieved chunks for debugging RAG."""
    chunks = rag_service.retrieve_chunks(request.query, limit=request.limit)
    return {"chunks": chunks}

@router.get("/documents")
async def list_documents():
    """Lists indexed papers."""
    return {"documents": indexed_documents}

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Removes a paper from the knowledge base."""
    if document_id not in indexed_documents:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    rag_service.qdrant.delete_document(document_id)
    del indexed_documents[document_id]
    
    # Also delete the file
    file_path = f"../datasets/uploads/{document_id}.pdf"
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return {"status": "deleted", "document_id": document_id}

@router.get("/model-info")
async def get_model_info():
    """Returns model, embedding, and quantization details."""
    return {
        "base_model": "Qwen/Qwen2.5-3B",
        "fine_tuning": "QLoRA on QASPER",
        "quantization": "AWQ (4-bit)",
        "embedding_model": "BAAI/bge-m3",
        "vector_db": "Qdrant",
        "inference_engine": "vLLM"
    }

@router.get("/health")
async def health_check():
    """Checks API and services status."""
    # Assuming basic health here. In production, check Qdrant and vLLM connection actively.
    return {"status": "healthy", "components": ["api", "qdrant", "vllm"]}
