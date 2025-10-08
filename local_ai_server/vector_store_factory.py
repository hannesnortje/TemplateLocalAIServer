"""
Factory for creating vector store instances, allowing easy switching
between different vector database implementations.
"""
import logging
from typing import Any, Optional
import os

from .config import VECTOR_DB_TYPE

logger = logging.getLogger(__name__)

# Default instances
_vector_store = None
_store_class = None
_get_store_func = None
_store_type = None

def get_vector_store(db_type: Optional[str] = None, storage_path: Optional[str] = None) -> Any:
    """Get a vector store instance of the specified type.
    
    Args:
        db_type: The type of vector database to use ('qdrant' or 'chroma')
                If None, uses the value from configuration
        storage_path: Optional custom storage path
        
    Returns:
        A vector store instance
    """
    global _vector_store, _store_class, _get_store_func, _store_type
    
    # If no type specified, use configured default or environment override
    selected_type = db_type or os.getenv("VECTOR_DB_TYPE", VECTOR_DB_TYPE)
    
    # Only re-initialize if the store type changes or we need a custom storage path
    if selected_type != _store_type or storage_path is not None or _vector_store is None:
        try:
            if selected_type.lower() == 'chroma':
                from .chroma_store import ChromaVectorStore, get_chroma_store
                _store_class = ChromaVectorStore
                _get_store_func = get_chroma_store
                logger.info(f"Using ChromaDB vector store (path: {storage_path or 'default'})")
            else:
                # Default to Qdrant
                from .vector_store import VectorStore, get_vector_store as get_qdrant_store
                _store_class = VectorStore
                _get_store_func = get_qdrant_store
                logger.info(f"Using Qdrant vector store (path: {storage_path or 'default'})")
            
            _store_type = selected_type
            
            # Create a new instance with custom path if needed
            if storage_path is not None:
                return _get_store_func(storage_path)
                
            # Return singleton instance
            _vector_store = _get_store_func()
            return _vector_store
            
        except ImportError as e:
            logger.error(f"Error importing vector store implementation: {e}")
            raise RuntimeError(f"Vector store type '{selected_type}' is not available: {e}")
    
    # Return existing instance
    return _vector_store
