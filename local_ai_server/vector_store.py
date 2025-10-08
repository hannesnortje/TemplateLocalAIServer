import logging
import os
import stat
import uuid
import tempfile
import time
import random
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from .config import (
    QDRANT_PATH, QDRANT_COLLECTION,
    VECTOR_SIZE, EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)

class VectorStore:
    _instance = None
    _lock_wait_time = 30  # Maximum time to wait for lock in seconds
    
    def __new__(cls, storage_path=None):
        if cls._instance is None or storage_path:  # Create new instance if specific path requested
            instance = super(VectorStore, cls).__new__(cls)
            instance.initialized = False
            if storage_path:
                # For testing: Don't use singleton with custom path
                return instance
            cls._instance = instance
        return cls._instance

    def __init__(self, storage_path=None):
        if hasattr(self, 'initialized') and not self.initialized or storage_path:
            self.storage_path = storage_path or QDRANT_PATH
            
            # Ensure directory exists with proper permissions
            logger.debug(f"Setting up vector store at {self.storage_path}")
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Set full permissions for testing
            try:
                for root, dirs, files in os.walk(self.storage_path):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o777)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o666)
                
                # Set directory permissions
                os.chmod(self.storage_path, 0o777)
            except Exception as e:
                logger.warning(f"Could not set permissions: {e}")
            
            # Initialize client with retry mechanism
            self._init_client_with_retry()
                
    def _init_client_with_retry(self, max_attempts=5):  # Increase max attempts
        """Initialize client with retry mechanism to handle locks"""
        attempts = 0
        last_error = None
        
        while attempts < max_attempts:
            try:
                # Close previous client if exists
                if hasattr(self, 'client') and self.client:
                    try:
                        self.client.close()
                    except:
                        pass
                
                # Try to initialize the client with exponential backoff
                wait_time = random.uniform(1, 3) * (2 ** attempts)  # Exponential with jitter
                if attempts > 0:
                    logger.warning(f"Retrying vector store initialization in {wait_time:.1f} seconds... ({attempts+1}/{max_attempts})")
                    time.sleep(wait_time)
                
                # Try a different approach: use memory only for tests
                if attempts >= 3 and str(self.storage_path).endswith("vectors"):
                    logger.warning("Trying with in-memory collection instead of persistent storage")
                    self.client = QdrantClient(":memory:")  # Use in-memory storage after several failures
                else:
                    # Normal initialization with persistent storage
                    self.client = QdrantClient(path=str(self.storage_path))
                
                self.model = SentenceTransformer(EMBEDDING_MODEL)
                self._ensure_collection()
                self.initialized = True
                
                if attempts > 0:
                    logger.info(f"Successfully initialized vector store after {attempts+1} attempts")
                return
            
            except Exception as e:
                attempts += 1
                last_error = e
                
                if "already accessed by another instance" in str(e):
                    # If it's a lock issue, wait a bit with random jitter and try again
                    wait_time = random.uniform(1, 3) * attempts
                    logger.warning(f"Vector storage is locked. Retrying in {wait_time:.1f} seconds... (Attempt {attempts}/{max_attempts})")
                    time.sleep(wait_time)
                else:
                    # For other errors, log and retry
                    logger.error(f"Error initializing vector store (attempt {attempts}/{max_attempts}): {e}")
                    time.sleep(1)
        
        # If we get here, all attempts failed
        logger.error(f"Failed to initialize vector store after {max_attempts} attempts. Last error: {last_error}")
        
        # Suggest solutions based on the error
        if "already accessed by another instance" in str(last_error):
            logger.error("SOLUTION: The vector database is locked by another process. Try:")
            logger.error("1. Stop all other instances of the server")
            logger.error("2. Check for zombie processes that might be holding the lock")
            logger.error("3. Delete the lock file in the storage directory")
            logger.error(f"4. If all else fails, you can delete and recreate the storage directory: {self.storage_path}")
        
        raise last_error

    def _ensure_collection(self):
        """Ensure the vector collection exists with retry mechanism"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.client.get_collection(QDRANT_COLLECTION)
                return
            except Exception as e:
                if "Collection not found" in str(e) or attempt == max_attempts - 1:
                    # Create collection on first attempt or last attempt
                    try:
                        self.client.create_collection(
                            collection_name=QDRANT_COLLECTION,
                            vectors_config=models.VectorParams(
                                size=VECTOR_SIZE,
                                distance=models.Distance.COSINE
                            )
                        )
                        logger.info(f"Created collection: {QDRANT_COLLECTION}")
                        return
                    except Exception as create_error:
                        if attempt < max_attempts - 1:
                            logger.warning(f"Error creating collection (attempt {attempt+1}): {create_error}")
                            time.sleep(1)
                        else:
                            raise

    def add_texts(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> List[str]:
        """Add texts to the vector store with retry mechanism"""
        if not self.initialized:
            self._init_client_with_retry()
            
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                points = []
                
                if metadata is None:
                    metadata = [{} for _ in texts]

                for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
                    point_id = uuid.uuid4().int % (2**63)
                    points.append(models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "text": text,
                            **meta
                        }
                    ))

                self.client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points
                )
                return [str(p.id) for p in points]
            except Exception as e:
                if "already accessed by another instance" in str(e):
                    # Re-initialize client and retry
                    logger.warning(f"Vector store locked during add_texts. Retrying... ({attempt+1}/{max_attempts})")
                    self._init_client_with_retry()
                elif attempt < max_attempts - 1:
                    logger.warning(f"Error adding texts (attempt {attempt+1}): {e}")
                    time.sleep(1)
                else:
                    logger.error(f"Failed to add texts after {max_attempts} attempts: {e}")
                    raise

    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar texts using vector similarity with retry mechanism"""
        if not self.initialized:
            self._init_client_with_retry()
            
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                vector = self.model.encode(query, convert_to_numpy=True)
                
                # Convert filter dictionary to proper Qdrant filter format
                query_filter = None
                if filter:
                    # Create proper filter condition
                    filter_conditions = []
                    for key, value in filter.items():
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
                    
                    if filter_conditions:
                        query_filter = models.Filter(
                            must=filter_conditions
                        )
                
                try:
                    # Try the new recommended method first
                    search_result = self.client.query_points(
                        collection_name=QDRANT_COLLECTION,
                        query_vector=vector.tolist(),
                        query_filter=query_filter,
                        limit=k
                    )
                except Exception as query_error:
                    # Fall back to deprecated method if new method fails
                    logger.warning(f"query_points failed, falling back to search: {query_error}")
                    search_result = self.client.search(
                        collection_name=QDRANT_COLLECTION,
                        query_vector=vector.tolist(),
                        limit=k,
                        query_filter=query_filter
                    )
                
                return [{
                    "text": hit.payload["text"],
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                    "score": hit.score
                } for hit in search_result]
                
            except Exception as e:
                if "already accessed by another instance" in str(e):
                    # Re-initialize client and retry
                    logger.warning(f"Vector store locked during similarity_search. Retrying... ({attempt+1}/{max_attempts})")
                    self._init_client_with_retry()
                elif attempt < max_attempts - 1:
                    logger.warning(f"Error during search (attempt {attempt+1}): {e}")
                    time.sleep(1)
                else:
                    logger.error(f"Failed similarity search after {max_attempts} attempts: {e}")
                    raise

    def delete_texts(self, ids: List[str]):
        """Delete texts by their IDs"""
        self.client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.PointIdsList(
                points=list(map(int, ids))
            )
        )

    def close(self):
        """Close the Qdrant client connection"""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error closing vector store client: {e}")
            delattr(self, 'client')
        self.initialized = False

    def __del__(self):
        """Cleanup when instance is deleted"""
        try:
            self.close()
        except:
            pass

# Don't create instance on import
vector_store = None

def get_vector_store(storage_path=None):
    """Get or create vector store instance"""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore(storage_path)
    return vector_store
