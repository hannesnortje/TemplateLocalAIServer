import unittest
import shutil
import tempfile
import os
from pathlib import Path
import json
import time
from local_ai_server.vector_store import VectorStore
from local_ai_server.config import VECTOR_STORAGE_DIR

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        """Set up test vector store with temporary storage"""
        # Create a temp directory with proper permissions
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qdrant_test_"))
        
        # Set permissions
        os.chmod(self.temp_dir, 0o777)
        
        # Create a fresh vector store instance for each test
        self.vector_store = VectorStore(storage_path=self.temp_dir)
        
        # Wait a moment for initialization
        time.sleep(0.1)

    def tearDown(self):
        """Clean up test data"""
        try:
            if hasattr(self, 'vector_store'):
                del self.vector_store
                
            # Force a small delay before cleanup
            time.sleep(0.5)
            
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                # Force permissions for cleanup
                for root, dirs, files in os.walk(self.temp_dir):
                    for d in dirs:
                        try:
                            os.chmod(os.path.join(root, d), 0o777)
                        except:
                            pass
                    for f in files:
                        try:
                            os.chmod(os.path.join(root, f), 0o666)
                        except:
                            pass
                
                # Remove directory
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
        self.assertTrue(all("cat" in r["metadata"]["animal"] for r in results))

    def test_persistence(self):
        """Test that vectors persist between instances"""
        # Add a document
        text = "This is a persistent test document"
        ids = self.vector_store.add_texts([text])
        
        # Create new instance
        del self.vector_store
        time.sleep(1)  # Wait for cleanup
        new_store = VectorStore(storage_path=self.temp_dir)
        
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

if __name__ == '__main__':
    unittest.main()
