import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

class QdrantManager:
    def __init__(self, host="localhost", port=6333, collection_name="papers"):
        if host == "localhost":
            # For local VS Code testing without Docker, use a local folder!
            self.client = QdrantClient(path="../local_qdrant_db")
        else:
            # When inside Docker, connect to the Qdrant container
            self.client = QdrantClient(host=host, port=port)
            
        self.collection_name = collection_name
        
        # BGE-M3 embedding size is 1024
        self.vector_size = 1024
        self._ensure_collection_exists()
        
    def _ensure_collection_exists(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            
    def insert_chunks(self, document_id: str, title: str, chunks_with_embeddings: list):
        """Inserts document chunks into Qdrant."""
        points = []
        for item in chunks_with_embeddings:
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=item["embedding"],
                    payload={
                        "document_id": document_id,
                        "title": title,
                        "text": item["text"]
                    }
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
    def search(self, query_embedding: list, limit: int = 5):
        """Searches for similar chunks."""
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit
        )
        return search_result.points
        
    def delete_document(self, document_id: str):
        """Deletes all chunks associated with a document_id."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={"filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]}}
        )
