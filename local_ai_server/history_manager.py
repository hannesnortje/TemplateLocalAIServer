import logging
import time
import uuid
import os
from pathlib import Path  # Add this import
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from .config import (
    QDRANT_PATH, VECTOR_SIZE, EMBEDDING_MODEL,
    RESPONSE_HISTORY_COLLECTION, ENABLE_RESPONSE_HISTORY,
    MAX_HISTORY_ITEMS, HISTORY_RETENTION_DAYS,
    VECTOR_DB_TYPE, CHROMA_PATH
)

logger = logging.getLogger(__name__)

class ResponseHistoryManager:
    """Manager for storing and retrieving response history in the vector store.
    
    This class handles saving responses, retrieving historical responses,
    and cleanup of old entries.
    """
    _instance = None
    
    def __new__(cls, storage_path=None):
        if cls._instance is None or storage_path:  # Allow custom path for testing
            instance = super(ResponseHistoryManager, cls).__new__(cls)
            instance.initialized = False
            if storage_path:
                # For testing: Don't use singleton with custom path
                return instance
            cls._instance = instance
        return cls._instance
    
    def __init__(self, storage_path=None):
        if not hasattr(self, 'initialized') or not self.initialized or storage_path:
            # Select appropriate storage path based on vector DB type
            if storage_path is None:
                if VECTOR_DB_TYPE.lower() == 'chroma':
                    # For ChromaDB storage, use a 'history' subdirectory to avoid conflicts
                    self.storage_path = CHROMA_PATH / 'history'
                else:
                    # For Qdrant, use the same path but different collection
                    self.storage_path = QDRANT_PATH
            else:
                self.storage_path = storage_path
            
            self.enabled = ENABLE_RESPONSE_HISTORY
            self.initialized = False  # Set to False until fully initialized
            self.using_chroma = VECTOR_DB_TYPE.lower() == 'chroma' or (storage_path and str(storage_path).endswith('chroma'))
            logger.debug(f"Preparing response history manager with storage at {self.storage_path} (using ChromaDB: {self.using_chroma})")
    
    def _initialize(self):
        """Delayed initialization to avoid lock conflicts."""
        if self.initialized:
            return
        
        try:
            # Log additional debugging info
            logger.debug(f"Initializing history manager with path: {self.storage_path}")
            logger.debug(f"Directory exists: {Path(self.storage_path).exists()}")
            
            # Ensure the directory exists with proper permissions
            os.makedirs(self.storage_path, exist_ok=True)
            try:
                os.chmod(self.storage_path, 0o777)
            except Exception as e:
                logger.warning(f"Could not set permissions: {e}")
            
            # Initialize vector store client - use ChromaDB if configured
            if self.using_chroma or (isinstance(self.storage_path, str) and self.storage_path == ":memory:"):
                # Use ChromaDB for history too
                logger.debug("Using ChromaDB for history")
                self.client = chromadb.PersistentClient(
                    path=str(self.storage_path),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                # Initialize embedding model
                self.model = SentenceTransformer(EMBEDDING_MODEL)
                self._ensure_chroma_collection()
            else:
                # Use Qdrant (legacy approach)
                logger.debug("Using Qdrant for history")
                self.client = QdrantClient(path=str(self.storage_path))
                self.model = SentenceTransformer(EMBEDDING_MODEL)
                self._ensure_qdrant_collection()
            
            # Force enabled for tests
            if isinstance(self.storage_path, str) and self.storage_path == ":memory:" or \
               (self.storage_path != QDRANT_PATH and self.storage_path != CHROMA_PATH / 'history'):
                logger.debug("Forcing history enabled for test instance")
                self.enabled = True
            
            self.initialized = True
            logger.info(f"Response history manager initialized (enabled: {self.enabled})")
        except Exception as e:
            logger.error(f"Failed to initialize history manager: {e}")
            raise
    
    def _ensure_qdrant_collection(self):
        """Ensure the Qdrant response history collection exists"""
        try:
            self.client.get_collection(RESPONSE_HISTORY_COLLECTION)
            logger.debug(f"Found existing Qdrant collection: {RESPONSE_HISTORY_COLLECTION}")
        except Exception:
            self.client.create_collection(
                collection_name=RESPONSE_HISTORY_COLLECTION,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {RESPONSE_HISTORY_COLLECTION}")
    
    def _ensure_chroma_collection(self):
        """Ensure the ChromaDB response history collection exists"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=RESPONSE_HISTORY_COLLECTION)
            logger.debug(f"Found existing ChromaDB collection: {RESPONSE_HISTORY_COLLECTION}")
        except Exception:
            # Create collection if it doesn't exist
            self.collection = self.client.create_collection(
                name=RESPONSE_HISTORY_COLLECTION,
                embedding_function=None,  # We'll handle embeddings ourselves
                metadata={"description": "Response history storage"}
            )
            logger.info(f"Created ChromaDB collection: {RESPONSE_HISTORY_COLLECTION}")
    
    def save_response(self, query: str, response: Any, metadata: Optional[Dict] = None) -> Optional[str]:
        """Save a response to the history.
        
        Args:
            query: The user's query
            response: The system's response
            metadata: Additional metadata about the response
            
        Returns:
            ID of the saved entry or None if history is disabled
        """
        if not self.initialized:
            logger.debug("Initializing history manager during save_response call")
            self._initialize()
            
        # Additional debug logging
        logger.debug(f"History enabled: {self.enabled}")
        
        if not self.enabled:
            logger.debug("History is disabled, not saving response")
            return None
        
        try:
            # Create embedding for the query
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Ensure we have metadata
            if metadata is None:
                metadata = {}
            
            # Ensure we have a timestamp
            if 'timestamp' not in metadata:
                metadata['timestamp'] = time.time()
                
            if self.using_chroma:
                # Using ChromaDB
                # Generate a unique ID
                point_id = str(uuid.uuid4())
                
                # Format metadata for Chroma
                chroma_metadata = {
                    "query": query,
                    "response": response,
                    **metadata  # Flatten metadata structure
                }
                
                # Add to ChromaDB collection
                self.collection.add(
                    ids=[point_id],
                    embeddings=[query_embedding.tolist()],
                    metadatas=[chroma_metadata]
                )
            else:
                # Using Qdrant
                # Generate a unique ID
                point_id = uuid.uuid4().int % (2**63)
                
                # Add the point to the collection
                self.client.upsert(
                    collection_name=RESPONSE_HISTORY_COLLECTION,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=query_embedding.tolist(),
                            payload={
                                "query": query,
                                "response": response,
                                "metadata": metadata
                            }
                        )
                    ]
                )
                point_id = str(point_id)
            
            logger.debug(f"Saved response to history with ID: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Error saving response to history: {e}")
            return None
    
    def find_similar_responses(
        self, 
        query: str, 
        limit: int = MAX_HISTORY_ITEMS,
        min_score: float = 0.7,
        filter_params: Optional[Dict] = None
    ) -> List[Dict]:
        """Find similar previous responses based on query similarity."""
        if not self.enabled:
            return []
            
        if not self.initialized:
            self._initialize()
            
        try:
            # For empty query, return most recent entries matching filter
            if not query.strip():
                return self._get_filtered_responses(filter_params, limit)
                
            # Generate embedding for query
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            if self.using_chroma:
                # Using ChromaDB
                # Format filter for Chroma
                where_document = None
                if filter_params:
                    where_document = filter_params
                
                # Search for similar responses
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(limit, 100),  # Avoid requesting more than available
                    where=where_document,
                    include=["metadatas", "documents", "distances"]
                )
                
                # Format results from ChromaDB format
                responses = []
                if results["ids"] and len(results["ids"][0]) > 0:
                    for i in range(len(results["ids"][0])):
                        metadata = results["metadatas"][0][i].copy()
                        # Extract query and response from metadata
                        query_text = metadata.pop("query", "")
                        response_text = metadata.pop("response", "")
                        
                        # Get similarity score (convert distance to similarity)
                        if "distances" in results and len(results["distances"]) > 0:
                            # ChromaDB uses cosine distance (0 is identical, 2 is opposite)
                            # Convert to similarity score (1 is identical, 0 is opposite)
                            distance = float(results["distances"][0][i])
                            score = 1.0 - (distance / 2.0)
                        else:
                            score = 0.9  # Default if distance not available
                        
                        # Only include results above min_score
                        if score >= min_score:
                            responses.append({
                                "id": results["ids"][0][i],
                                "query": query_text,
                                "response": response_text,
                                "metadata": metadata,
                                "similarity": score
                            })
                
                # If no results and we have a filter, try without it
                if not responses and filter_params:
                    # Try again without filter
                    logger.debug("No results with filter, trying without filter")
                    basic_results = self.collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=min(limit, 100),
                        include=["metadatas", "documents", "distances"]
                    )
                    
                    if basic_results["ids"] and len(basic_results["ids"][0]) > 0:
                        for i in range(len(basic_results["ids"][0])):
                            metadata = basic_results["metadatas"][0][i].copy()
                            # Extract query and response
                            query_text = metadata.pop("query", "")
                            response_text = metadata.pop("response", "")
                            
                            # Calculate score
                            if "distances" in basic_results and len(basic_results["distances"]) > 0:
                                distance = float(basic_results["distances"][0][i])
                                score = 1.0 - (distance / 2.0)
                            else:
                                score = 0.9
                            
                            if score >= min_score:
                                responses.append({
                                    "id": basic_results["ids"][0][i],
                                    "query": query_text,
                                    "response": response_text,
                                    "metadata": metadata,
                                    "similarity": score
                                })
                                
                                if len(responses) >= limit:
                                    break
            else:
                # Using Qdrant
                # Set up filter 
                query_filter = None
                if filter_params:
                    filter_conditions = []
                    for key, value in filter_params.items():
                        filter_conditions.append(
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchValue(value=value)
                            )
                        )
                    
                    if filter_conditions:
                        query_filter = models.Filter(
                            must=filter_conditions
                        )
                
                # Search for similar responses
                results = self.client.search(
                    collection_name=RESPONSE_HISTORY_COLLECTION,
                    query_vector=query_embedding.tolist(),
                    limit=limit,
                    query_filter=query_filter,
                    score_threshold=min_score
                )
                
                # Format results
                responses = []
                for hit in results:
                    responses.append({
                        "query": hit.payload["query"],
                        "response": hit.payload["response"],
                        "metadata": hit.payload["metadata"],
                        "similarity": hit.score
                    })
        
            logger.debug(f"Found {len(responses)} similar historical responses")
            return responses
            
        except Exception as e:
            logger.error(f"Error finding similar responses: {e}")
            return []

    def _get_filtered_responses(self, filter_params: Optional[Dict] = None, limit: int = 10) -> List[Dict]:
        """Get most recent responses matching the filter."""
        try:
            if self.using_chroma:
                # For ChromaDB, get all entries matching filter or all entries if no filter
                where_filter = filter_params if filter_params else None
                
                results = self.collection.get(
                    where=where_filter,
                    limit=limit
                )
                
                responses = []
                if results and "ids" in results and len(results["ids"]) > 0:
                    for i in range(len(results["ids"])):
                        metadata = results["metadatas"][i].copy()
                        # Extract query and response
                        query_text = metadata.pop("query", "")
                        response_text = metadata.pop("response", "")
                        
                        responses.append({
                            "id": results["ids"][i],
                            "query": query_text,
                            "response": response_text,
                            "metadata": metadata,
                            "similarity": 1.0  # Default high similarity as these are direct matches
                        })
                    
                    # Sort by timestamp if available
                    responses.sort(
                        key=lambda x: x.get("metadata", {}).get("timestamp", 0),
                        reverse=True  # Most recent first
                    )
                    
                    # Limit to requested number
                    responses = responses[:limit]
                
                return responses
            else:
                # Qdrant implementation
                # ...existing code for Qdrant...
                return []
                
        except Exception as e:
            logger.error(f"Error getting filtered responses: {e}")
            return []

    def clean_old_entries(self, days: int = HISTORY_RETENTION_DAYS) -> int:
        """Remove entries older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of entries removed
        """
        if not self.enabled:
            return 0
            
        if not self.initialized:
            self._initialize()
            
        try:
            # Calculate cutoff timestamp
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            
            if self.using_chroma:
                # Using ChromaDB
                # First, get all IDs
                all_ids = self.collection.get()["ids"]
                
                # Then get metadata for all entries
                if all_ids:
                    metadata = self.collection.get(ids=all_ids)["metadatas"]
                    
                    # Find IDs to delete
                    ids_to_delete = []
                    for i, meta in enumerate(metadata):
                        timestamp = meta.get("timestamp", 0)
                        if timestamp < cutoff_time:
                            ids_to_delete.append(all_ids[i])
                    
                    # Delete old entries
                    if ids_to_delete:
                        self.collection.delete(ids=ids_to_delete)
                        logger.info(f"Cleaned up {len(ids_to_delete)} old history entries")
                        return len(ids_to_delete)
                
                return 0
            else:
                # Using Qdrant
                # Create filter to find old entries
                old_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.timestamp",
                            range=models.Range(
                                lt=cutoff_time
                            )
                        )
                    ]
                )
                
                # Search for old entries to get count
                count_result = self.client.count(
                    collection_name=RESPONSE_HISTORY_COLLECTION,
                    count_filter=old_filter
                )
                
                if count_result.count == 0:
                    logger.debug("No old entries to clean up")
                    return 0
                    
                # Delete old entries
                self.client.delete(
                    collection_name=RESPONSE_HISTORY_COLLECTION,
                    points_selector=models.FilterSelector(
                        filter=old_filter
                    )
                )
                
                logger.info(f"Cleaned up {count_result.count} old history entries")
                return count_result.count
            
        except Exception as e:
            logger.error(f"Error cleaning old entries: {e}")
            return 0
    
    def delete_all_history(self) -> bool:
        """Delete all history entries.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        if not self.initialized:
            self._initialize()
            
        try:
            if self.using_chroma:
                # Using ChromaDB
                # Get all IDs
                all_ids = self.collection.get()["ids"]
                
                # Delete all entries
                if all_ids:
                    self.collection.delete(ids=all_ids)
                
                logger.info("Deleted all response history (ChromaDB)")
            else:
                # Using Qdrant
                # Recreate the collection
                self.client.delete_collection(RESPONSE_HISTORY_COLLECTION)
                self._ensure_qdrant_collection()
                
                logger.info("Deleted all response history (Qdrant)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting all history: {e}")
            return False

    def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get all conversation history for a specific user.
        
        Args:
            user_id: The user's unique identifier
            limit: Maximum number of history items to return
            
        Returns:
            List of conversations with this user
        """
        if not self.enabled:
            return []
            
        if not self.initialized:
            self._initialize()
            
        try:
            if self.using_chroma:
                # Using ChromaDB
                # For ChromaDB, we use a where filter to get all items for this user
                results = self.collection.get(
                    where={"user_id": user_id},
                    limit=limit
                )
                
                # Format results
                responses = []
                if results and "ids" in results and len(results["ids"]) > 0:
                    for i in range(len(results["ids"])):
                        metadata = results["metadatas"][i].copy()
                        # Extract query and response
                        query_text = metadata.pop("query", "")
                        response_text = metadata.pop("response", "")
                        
                        # Add to results
                        responses.append({
                            "id": results["ids"][i],
                            "query": query_text,
                            "response": response_text,
                            "metadata": metadata
                        })
                        
                    # Sort by timestamp
                    responses.sort(key=lambda x: x.get("metadata", {}).get("timestamp", 0))
                    
                return responses
            else:
                # Using Qdrant
                # For Qdrant, we create a filter for the user_id field
                user_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.user_id",
                            match=models.MatchValue(value=user_id)
                        )
                    ]
                )
                
                # Get points matching the filter
                results = self.client.scroll(
                    collection_name=RESPONSE_HISTORY_COLLECTION,
                    scroll_filter=user_filter,
                    limit=limit
                )
                
                # Format results
                responses = []
                for point in results[0]:
                    responses.append({
                        "id": str(point.id),
                        "query": point.payload["query"],
                        "response": point.payload["response"],
                        "metadata": point.payload["metadata"]
                    })
                    
                # Sort by timestamp
                responses.sort(key=lambda x: x.get("metadata", {}).get("timestamp", 0))
                
                return responses
                
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return []

    def close(self):
        """Close the vector store connection"""
        if self.using_chroma:
            if hasattr(self, 'collection'):
                self.collection = None
            if hasattr(self, 'client'):
                self.client = None
        else:
            if hasattr(self, 'client'):
                self.client.close()
                delattr(self, 'client')
        self.initialized = False

    def __del__(self):
        """Cleanup when instance is deleted"""
        try:
            self.close()
        except:
            pass

# Create a lazy-loading global instance instead of initializing immediately
response_history = ResponseHistoryManager()

def get_response_history(storage_path=None):
    """Get or create response history manager instance."""
    global response_history
    if storage_path:
        # For testing - create a new instance with custom path
        # Close existing instance if any
        if response_history and hasattr(response_history, 'client'):
            try:
                response_history.close()
            except:
                pass
        return ResponseHistoryManager(storage_path=storage_path)
    
    # Initialize if needed
    if not response_history.initialized:
        response_history._initialize()
    return response_history

def create_dummy_history_manager():
    """Create a minimal dummy history manager for fallback when real one fails"""
    class DummyHistoryManager:
        """Simple in-memory history manager that doesn't use Qdrant"""
        def __init__(self):
            self.history = []
            self.enabled = True
            self.initialized = True
            logger.warning("Using dummy history manager with limited functionality")
            
        def save_response(self, query, response, metadata=None):
            """Save response to in-memory list"""
            if not self.enabled:
                return None
            if metadata is None:
                metadata = {"timestamp": time.time()}
            self.history.append({"query": query, "response": response, "metadata": metadata})
            return "dummy-id"
            
        def find_similar_responses(self, query, limit=5, min_score=0.7, filter_params=None):
            """Simple keyword match - no vector similarity"""
            if not self.enabled:
                return []
            
            results = []
            for item in self.history:
                if any(word in item["query"].lower() for word in query.lower().split()):
                    # Apply filter if provided
                    if filter_params:
                        skip = False
                        for key, value in filter_params.items():
                            if item["metadata"].get(key) != value:
                                skip = True
                                break
                        if skip:
                            continue
                    
                    # Add to results with dummy similarity
                    results.append({
                        "query": item["query"],
                        "response": item["response"],
                        "metadata": item["metadata"],
                        "similarity": 0.9  # Dummy similarity score
                    })
                    
                    if len(results) >= limit:
                        break
                        
            return results
            
        def clean_old_entries(self, days=30):
            """Remove old entries based on timestamp"""
            if not self.enabled:
                return 0
                
            cutoff = time.time() - (days * 24 * 60 * 60)
            count_before = len(self.history)
            
            self.history = [
                item for item in self.history 
                if item["metadata"].get("timestamp", 0) >= cutoff
            ]
            
            return count_before - len(self.history)
            
        def delete_all_history(self):
            """Clear all history"""
            if not self.enabled:
                return False
                
            self.history = []
            return True
            
        def close(self):
            """No-op for dummy manager"""
            pass
    
    return DummyHistoryManager()
