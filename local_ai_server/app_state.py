"""
Module to hold shared application state and avoid circular imports.
This module initializes and holds global instances for vector store and history manager.
"""
import logging
import atexit
import time
import random
import os

# Configure logging
logger = logging.getLogger(__name__)

# Initialize placeholders for global objects
vector_store = None
history_manager = None
model_manager = None

def initialize_resources():
    """Initialize shared resources on demand"""
    global vector_store, history_manager
    
    # First, ensure storage directories exist with correct permissions
    try:
        from .config import QDRANT_PATH, CHROMA_PATH, VECTOR_DB_TYPE
        
        # Prepare paths based on selected vector db type
        if VECTOR_DB_TYPE.lower() == 'qdrant':
            store_path = QDRANT_PATH
        else:
            # ChromaDB is now the default
            store_path = CHROMA_PATH
            
        os.makedirs(store_path, exist_ok=True)
        os.chmod(store_path, 0o777)  # Set full permissions
        
        # Check for lock file and remove if needed for Qdrant
        if VECTOR_DB_TYPE.lower() == 'qdrant':
            lock_file = os.path.join(store_path, '.lock')
            if os.path.exists(lock_file):
                logger.warning(f"Found stale lock file at {lock_file}. Removing it...")
                try:
                    os.remove(lock_file)
                    time.sleep(1)  # Wait for file system
                except Exception as e:
                    logger.error(f"Could not remove lock file: {e}")
    except Exception as e:
        logger.error(f"Error preparing storage directories: {e}")
    
    # Try to initialize vector store first
    try:
        if vector_store is None:
            from .vector_store_factory import get_vector_store
            vector_store = get_vector_store()  # Uses configured type
            logger.info("Vector store initialized")
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
    
    # Use separate try block for history manager so vector store failure doesn't block it
    try:
        if history_manager is None:
            # Add jitter to avoid racing
            time.sleep(random.uniform(0.5, 1.5))
            
            # Setup in-memory history if vector store initialization failed
            if vector_store is None:
                from .history_manager import ResponseHistoryManager
                logger.warning("Using in-memory history manager due to vector store failure")
                history_manager = ResponseHistoryManager(storage_path=":memory:")
            else:
                from .history_manager import get_response_history
                history_manager = get_response_history()
                
            logger.info("History manager initialized")
    except Exception as e:
        logger.error(f"Error initializing history manager: {e}")
        
        # Create a fallback dummy history manager
        try:
            from .history_manager import create_dummy_history_manager
            history_manager = create_dummy_history_manager()
            logger.warning("Using dummy history manager due to initialization failure")
        except Exception as fallback_error:
            logger.error(f"Failed to create dummy history manager: {fallback_error}")
    
    return vector_store is not None and history_manager is not None

def cleanup_resources():
    """Clean up resources on application shutdown"""
    global vector_store, history_manager
    
    logger.info("Cleaning up application resources...")
    
    # Clean up vector store
    if vector_store:
        try:
            if hasattr(vector_store, 'close'):
                vector_store.close()
                logger.info("Vector store closed")
        except Exception as e:
            logger.error(f"Error closing vector store: {e}")
    
    # Clean up history manager
    if history_manager:
        try:
            if hasattr(history_manager, 'close'):
                history_manager.close()
                logger.info("History manager closed")
        except Exception as e:
            logger.error(f"Error closing history manager: {e}")

# Register cleanup function to be called at exit
atexit.register(cleanup_resources)

# Initialize resources on module import
initialize_resources()
