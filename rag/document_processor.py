import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self, embedding_model_name="BAAI/bge-m3"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts raw text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        """Splits text into smaller chunks with overlap."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def get_embeddings(self, chunks: list) -> list:
        """Generates embeddings for a list of text chunks."""
        embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True)
        return embeddings.tolist()
        
    def process_document(self, pdf_path: str):
        """End-to-end processing: extract, chunk, embed."""
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        embeddings = self.get_embeddings(chunks)
        
        return [{"text": chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)]
