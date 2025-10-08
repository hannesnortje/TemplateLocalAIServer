import unittest
import json
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from local_ai_server.config import QDRANT_PATH
from local_ai_server.vector_store import VectorStore

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create temporary directory for vector storage"""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.original_path = QDRANT_PATH
        
        # Override the path before any imports that might use it
        import local_ai_server.config as config
        config.QDRANT_PATH = cls.temp_dir
        
        # Import and create Flask test client
        from local_ai_server.server import app
        cls.flask_app = app
        cls.flask_app.config['TESTING'] = True
        cls.client = cls.flask_app.test_client()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        import local_ai_server.config as config
        config.QDRANT_PATH = cls.original_path
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up test case"""
        self.app = self.__class__.client

    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')

    def test_list_models(self):
        response = self.app.get('/v1/models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIsInstance(data['data'], list)

    def test_available_models(self):
        response = self.app.get('/api/available-models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, dict)
        # Check for required model fields
        for model_id, model_info in data.items():
            self.assertIn('name', model_info)
            self.assertIn('url', model_info)
            self.assertIn('type', model_info)

    def test_add_documents(self):
        """Test adding documents to vector store"""
        response = self.app.post('/api/documents', json={
            "texts": ["This is a test document", "Another test document"],
            "metadata": [{"source": "test1"}, {"source": "test2"}]
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('ids', data)
        self.assertEqual(len(data['ids']), 2)
        self.ids = data['ids']  # Store ids as instance variable instead of returning

    def test_search_documents(self):
        """Test searching documents"""
        response = self.app.post('/api/search', json={
            "query": "test document",
            "limit": 2
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('results', data)
        self.assertTrue(len(data['results']) <= 2)

    def test_delete_documents(self):
        """Test deleting documents"""
        self.test_add_documents()  # Add documents and store ids
        response = self.app.delete('/api/documents', json={"ids": self.ids})
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
