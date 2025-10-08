"""
ChromaDB implementation of vector store.
This is a more concurrency-friendly alternative to Qdrant.
"""
import logging
import os
import time
import uuid
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer

from .config import (
    CHROMA_PATH, CHROMA_COLLECTION,
    VECTOR_SIZE, EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Vector store implementation using ChromaDB."""
    
    _instance = None
    
    def __new__(cls, storage_path=None):
        if cls._instance is None or storage_path:  # Create new instance if specific path requested
            instance = super(ChromaVectorStore, cls).__new__(cls)
            instance.initialized = False
            if storage_path:
                # For testing: Don't use singleton with custom path
                return instance
            cls._instance = instance
        return cls._instance
    
    def __init__(self, storage_path=None):
        """Initialize the ChromaDB vector store."""
        if not hasattr(self, 'initialized') or not self.initialized or storage_path:
            self.storage_path = storage_path or CHROMA_PATH
            self.collection_name = CHROMA_COLLECTION
            self.initialized = False
            
            # Ensure storage directory exists
            os.makedirs(self.storage_path, exist_ok=True)
            logger.debug(f"Setting up Chroma store at {self.storage_path}")
            
            # Initialize right away (Chroma handles concurrency better)
            self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        if self.initialized:
            return
            
        try:
            # Initialize ChromaDB with persistence
            self.client = chromadb.PersistentClient(
                path=str(self.storage_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Try to get existing collection or create a new one
            try:
                # First try to get existing collection
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.debug(f"Found existing collection: {self.collection_name}")
            except chromadb.errors.InvalidCollectionException:
                # If collection doesn't exist, create it
                logger.debug(f"Collection doesn't exist, creating new: {self.collection_name}")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,  # We'll handle embeddings ourselves
                    metadata={"description": "Document storage"}
                )
                logger.info(f"Created collection: {self.collection_name}")
            
            # Initialize embedding model
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            
            self.initialized = True
            logger.info(f"ChromaDB vector store initialized at {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_texts(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> List[str]:
        """Add texts to the vector store with their embeddings."""
        if not self.initialized:
            self._initialize()
            
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Generate IDs
            ids = [str(uuid.uuid4()) for _ in texts]
            
            # Use provided metadata or create empty dicts
            if metadata is None:
                metadata = [{} for _ in texts]
            
            # Format metadata for Chroma (must be JSON serializable)
            formatted_metadata = []
            for i, meta in enumerate(metadata):
                # Add the text as part of metadata so we can retrieve it
                meta_copy = meta.copy()
                meta_copy["text"] = texts[i]
                formatted_metadata.append(meta_copy)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=[e.tolist() for e in embeddings],
                metadatas=formatted_metadata
            )
            
            return ids
        except Exception as e:
            logger.error(f"Error adding texts to ChromaDB: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar texts using vector similarity."""
        if not self.initialized:
            self._initialize()
            
        try:
            # Generate embedding for query
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Format filter for Chroma
            where_document = None
            if filter:
                where_document = filter
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where_document
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    metadata = results["metadatas"][0][i]
                    # Extract text from metadata
                    text = metadata.pop("text", "")
                    
                    formatted_results.append({
                        "text": text,
                        "metadata": metadata,
                        "score": float(results["distances"][0][i]) if "distances" in results else 1.0
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching in ChromaDB: {e}")
            raise
    
    def delete_texts(self, ids: List[str]):
        """Delete texts by their IDs."""
        if not self.initialized:
            self._initialize()
            
        try:
            self.collection.delete(ids=ids)
        except Exception as e:
            logger.error(f"Error deleting texts from ChromaDB: {e}")
            raise
    
    def close(self):
        """Close ChromaDB client connection."""
        if hasattr(self, 'client'):
            # ChromaDB doesn't have an explicit close method, but we'll
            # dereference the objects to let Python handle the cleanup
            self.collection = None
            self.client = None
            self.initialized = False
    
    def __del__(self):
        """Cleanup when instance is deleted."""
        try:
            self.close()
        except:
            pass

# Don't create instance on import
chroma_store = None

def get_chroma_store(storage_path=None):
    """Get or create ChromaDB vector store instance."""
    global chroma_store
    if (chroma_store is None or storage_path):
        chroma_store = ChromaVectorStore(storage_path)
    return chroma_store
