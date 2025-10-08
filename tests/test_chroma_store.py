import unittest
import shutil
import tempfile
import os
from pathlib import Path
import json
import time
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Update config to use ChromaDB
import local_ai_server.config as config
config.VECTOR_DB_TYPE = "chroma"

from local_ai_server.chroma_store import ChromaVectorStore, get_chroma_store

class TestChromaStore(unittest.TestCase):
    def setUp(self):
        """Set up test vector store with temporary storage"""
        # Create a temp directory with proper permissions
        self.temp_dir = Path(tempfile.mkdtemp(prefix="chroma_test_"))
        
        # Set permissions
        os.chmod(self.temp_dir, 0o777)
        
        # Create a fresh vector store instance for each test
        self.vector_store = ChromaVectorStore(storage_path=self.temp_dir)
        
        # Wait a moment for initialization
        time.sleep(0.1)

    def tearDown(self):
        """Clean up test data"""
        try:
            if hasattr(self, 'vector_store'):
                self.vector_store.close()
                del self.vector_store
                
            # Force a small delay before cleanup
            time.sleep(0.5)
            
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Cleanup error (this can be ignored): {e}")

    def test_add_and_search(self):
        """Test adding documents and searching"""
        # Add test documents
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A quick brown cat sleeps on the windowsill",
            "The lazy dog sleeps in the sun"
        ]
        metadata = [
            {"source": "test1", "animal": "fox"},
            {"source": "test2", "animal": "cat"},
            {"source": "test3", "animal": "dog"}
        ]
        
        ids = self.vector_store.add_texts(texts, metadata)
        self.assertEqual(len(ids), 3)

        # Test basic search
        results = self.vector_store.similarity_search("quick animal", k=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(any("fox" in r["text"] for r in results))

        # Test filtered search
        results = self.vector_store.similarity_search(
            "sleeping animal",
            filter={"animal": "cat"}
        )
        self.assertTrue(all(r["metadata"]["animal"] == "cat" for r in results))

    def test_persistence(self):
        """Test that vectors persist between instances"""
        # Add a document
        text = "This is a persistent test document"
        ids = self.vector_store.add_texts([text])
        
        # Create new instance
        self.vector_store.close()
        del self.vector_store
        time.sleep(0.5)  # Wait for cleanup
        new_store = ChromaVectorStore(storage_path=self.temp_dir)
        
        # Search with new instance
        results = new_store.similarity_search("persistent test")
        self.assertTrue(len(results) > 0)
        self.assertTrue(any(text in r["text"] for r in results))

    def test_delete(self):
        """Test document deletion"""
        # Add documents
        texts = ["Document to keep", "Document to delete"]
        ids = self.vector_store.add_texts(texts)
        
        # Delete one document
        self.vector_store.delete_texts([ids[1]])
        
        # Verify deletion
        results = self.vector_store.similarity_search("document", k=10)
        self.assertTrue(any("keep" in r["text"] for r in results))
        self.assertFalse(any("delete" in r["text"] for r in results))

    def test_concurrency(self):
        """Basic test for concurrent access - just verifies no errors"""
        import threading
        
        def add_and_search():
            # Create a new connection to the same storage
            store = ChromaVectorStore(storage_path=self.temp_dir)
            # Add a document
            store.add_texts([f"Thread document {threading.get_ident()}"])
            # Search
            store.similarity_search("thread")
            # Close 
            store.close()
        
        # Create and start threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=add_and_search)
            threads.append(t)
            t.start()
            
        # Wait for all threads to complete
        for t in threads:
            t.join()
            
        # Verify we can still access the store
        results = self.vector_store.similarity_search("thread")
        self.assertTrue(len(results) > 0)

if __name__ == '__main__':
    unittest.main()
