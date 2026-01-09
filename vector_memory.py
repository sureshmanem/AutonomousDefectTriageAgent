"""
Vector Memory Module for Defect Triage System.

This module provides FAISS-based vector storage and similarity search
for log chunks using sentence-transformers embeddings.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from log_ingestor import LogChunk


@dataclass
class SearchResult:
    """Represents a similarity search result."""
    
    chunk: LogChunk
    score: float
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return (
            f"SearchResult(score={self.score:.4f}, "
            f"lines {self.chunk.line_start}-{self.chunk.line_end})"
        )


class VectorMemory:
    """
    FAISS-based vector memory for storing and searching log chunks.
    
    Uses sentence-transformers for local embeddings and FAISS for
    efficient similarity search.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "flat",
        dimension: Optional[int] = None
    ) -> None:
        """
        Initialize the VectorMemory.
        
        Args:
            model_name: HuggingFace model name for embeddings
            index_type: FAISS index type ('flat', 'ivf', or 'hnsw')
            dimension: Embedding dimension (auto-detected if None)
        """
        self.model_name = model_name
        self.index_type = index_type
        
        # Initialize sentence transformer
        print(f"Loading embedding model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        
        # Get embedding dimension
        self.dimension = dimension or self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = self._create_faiss_index()
        
        # Store original chunks and metadata
        self.chunks: List[LogChunk] = []
        self.metadata_store: List[Dict[str, Any]] = []
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _create_faiss_index(self) -> faiss.Index:
        """
        Create a FAISS index based on the specified type.
        
        Returns:
            FAISS index instance
        """
        if self.index_type == "flat":
            # L2 distance (Euclidean)
            return faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "ivf":
            # IVF with 100 clusters for faster search on large datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World for fast approximate search
            return faiss.IndexHNSWFlat(self.dimension, 32)
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, dimension])
        """
        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10
        )
        return embeddings.astype('float32')
    
    async def _encode_texts_async(self, texts: List[str]) -> np.ndarray:
        """
        Asynchronously encode texts into embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self._encode_texts,
            texts
        )
        return embeddings
    
    def add_documents(
        self,
        chunks: List[LogChunk],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Add log chunks to the vector database.
        
        Args:
            chunks: List of LogChunk objects to add
            metadata: Optional metadata for each chunk
            
        Returns:
            Number of documents added
        """
        if not chunks:
            return 0
        
        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self._encode_texts(texts)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            print("Training FAISS index...")
            self.index.train(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        
        if metadata:
            self.metadata_store.extend(metadata)
        else:
            self.metadata_store.extend([{}] * len(chunks))
        
        print(f"Added {len(chunks)} documents. Total: {len(self.chunks)}")
        return len(chunks)
    
    async def add_documents_async(
        self,
        chunks: List[LogChunk],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Asynchronously add log chunks to the vector database.
        
        Args:
            chunks: List of LogChunk objects to add
            metadata: Optional metadata for each chunk
            
        Returns:
            Number of documents added
        """
        if not chunks:
            return 0
        
        texts = [chunk.content for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks (async)...")
        embeddings = await self._encode_texts_async(texts)
        
        # Train and add (in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._add_to_index,
            embeddings
        )
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        
        if metadata:
            self.metadata_store.extend(metadata)
        else:
            self.metadata_store.extend([{}] * len(chunks))
        
        print(f"Added {len(chunks)} documents. Total: {len(self.chunks)}")
        return len(chunks)
    
    def _add_to_index(self, embeddings: np.ndarray) -> None:
        """Helper method to add embeddings to index."""
        if self.index_type == "ivf" and not self.index.is_trained:
            print("Training FAISS index...")
            self.index.train(embeddings)
        self.index.add(embeddings)
    
    def search_similar(
        self,
        query_text: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar log chunks using semantic similarity.
        
        Args:
            query_text: The query text (e.g., new error log)
            top_k: Number of top results to return
            score_threshold: Minimum similarity score (lower is better for L2)
            
        Returns:
            List of SearchResult objects, sorted by similarity
        """
        if len(self.chunks) == 0:
            return []
        
        # Encode query
        query_embedding = self._encode_texts([query_text])
        
        # Search FAISS index
        # Note: FAISS returns L2 distances (lower is better)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results: List[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            # Apply threshold if specified
            if score_threshold is not None and dist > score_threshold:
                continue
            
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(dist),
                index=int(idx),
                metadata=self.metadata_store[idx]
            ))
        
        return results
    
    async def search_similar_async(
        self,
        query_text: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Asynchronously search for similar log chunks.
        
        Args:
            query_text: The query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects
        """
        if len(self.chunks) == 0:
            return []
        
        # Encode query asynchronously
        query_embedding = await self._encode_texts_async([query_text])
        
        # Search in executor
        loop = asyncio.get_event_loop()
        distances, indices = await loop.run_in_executor(
            self.executor,
            lambda: self.index.search(query_embedding, top_k)
        )
        
        # Build results
        results: List[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            if score_threshold is not None and dist > score_threshold:
                continue
            
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(dist),
                index=int(idx),
                metadata=self.metadata_store[idx]
            ))
        
        return results
    
    def save(self, path: str | Path) -> None:
        """
        Save the vector memory to disk.
        
        Args:
            path: Directory path to save the memory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save chunks and metadata
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata_store, f)
        
        # Save config
        config = {
            "model_name": self.model_name,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "num_documents": len(self.chunks)
        }
        
        with open(path / "config.pkl", "wb") as f:
            pickle.dump(config, f)
        
        print(f"Saved vector memory to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> "VectorMemory":
        """
        Load vector memory from disk.
        
        Args:
            path: Directory path containing saved memory
            
        Returns:
            VectorMemory instance
        """
        path = Path(path)
        
        # Load config
        with open(path / "config.pkl", "rb") as f:
            config = pickle.load(f)
        
        # Create instance
        instance = cls(
            model_name=config["model_name"],
            index_type=config["index_type"],
            dimension=config["dimension"]
        )
        
        # Load FAISS index
        instance.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load chunks and metadata
        with open(path / "chunks.pkl", "rb") as f:
            instance.chunks = pickle.load(f)
        
        with open(path / "metadata.pkl", "rb") as f:
            instance.metadata_store = pickle.load(f)
        
        print(f"Loaded vector memory from {path}")
        print(f"Documents: {len(instance.chunks)}")
        
        return instance
    
    def clear(self) -> None:
        """Clear all data from the vector memory."""
        self.index.reset()
        self.chunks.clear()
        self.metadata_store.clear()
        print("Vector memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector memory.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "model_name": self.model_name,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "num_documents": len(self.chunks),
            "index_total": self.index.ntotal
        }
    
    def __len__(self) -> int:
        """Return the number of documents in memory."""
        return len(self.chunks)
    
    def __repr__(self) -> str:
        """String representation of VectorMemory."""
        return (
            f"VectorMemory(model='{self.model_name}', "
            f"index_type='{self.index_type}', "
            f"documents={len(self.chunks)})"
        )


# Example usage
async def main() -> None:
    """Example usage of VectorMemory with async operations."""
    from log_ingestor import LogIngestor
    
    # Create sample logs
    sample_logs = [
        """
        ERROR: Database connection failed
        java.sql.SQLException: Connection timeout
        at com.db.ConnectionPool.getConnection(ConnectionPool.java:123)
        Root cause: Network unreachable
        """,
        """
        NullPointerException in payment service
        java.lang.NullPointerException: Cannot read property 'amount'
        at com.payment.Service.process(Service.java:45)
        User ID: 12345
        """,
        """
        OutOfMemoryError during batch processing
        java.lang.OutOfMemoryError: Java heap space
        at com.batch.Processor.loadData(Processor.java:89)
        Heap size: 2GB, Required: 4GB
        """
    ]
    
    # Ingest logs
    ingestor = LogIngestor(chunk_size=50)
    all_chunks = []
    
    for log in sample_logs:
        chunks = ingestor.process_log_string(log)
        all_chunks.extend(chunks)
    
    print(f"\nProcessed {len(all_chunks)} chunks from sample logs\n")
    
    # Create vector memory
    memory = VectorMemory(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Add documents asynchronously
    await memory.add_documents_async(all_chunks)
    
    print(f"\n{memory}\n")
    
    # Search for similar errors
    query = "Database connection error with timeout"
    
    print(f"Searching for: '{query}'\n")
    results = await memory.search_similar_async(query, top_k=3)
    
    print(f"Found {len(results)} similar defects:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")
        print(f"   Content preview: {result.chunk.content[:150]}...")
        print()
    
    # Save to disk
    memory.save("./vector_db")
    
    # Load from disk
    loaded_memory = VectorMemory.load("./vector_db")
    print(f"\n{loaded_memory}")


if __name__ == "__main__":
    asyncio.run(main())
