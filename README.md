# Local AI Server

A self-hosted server that provides OpenAI-compatible APIs for running local language models with RAG capabilities.

## Features

- ✅ OpenAI-compatible API endpoints (drop-in replacement for applications)
- ✅ Support for GGUF format models (Llama, Mistral, Phi, etc.)
- ✅ Document storage with vector embeddings
- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Response history and conversational memory
- ✅ Swagger UI documentation
- ✅ Easy model management (download, upload, delete)
- ✅ HTTP and HTTPS support with automatic certificate generation

## Installation

### Prerequisites

- sudo apt update
- sudo apt install build-essential python3-dev cmake
- sudo apt install pkg-config libssl-dev
- Python 3.8 or higher
- pip or pipx (recommended)

### Install with pipx (recommended)

```bash
pipx install git+https://github.com/hannesnortje/LocalAIServer.git
```

This will install the package in an isolated environment and make the CLI command available globally.

### Install with pip

```bash
pip install git+https://github.com/hannesnortje/LocalAIServer.git
```

### GPU Support (optional)

For GPU acceleration, install with CUDA support:

```bash
pip install "git+https://github.com/hannesnortje/LocalAIServer.git#egg=local_ai_server[cuda]"
```

## Quick Start

### Start the server

```bash
local-ai-server start
```

The server will start on:
- HTTP: http://localhost:5000
- HTTPS: https://localhost:5443
- API Documentation: http://localhost:5000/docs
- Web UI: http://localhost:5000

### Download or Upload a Model

#### Using the Web Interface
1. Open http://localhost:5000 in your browser
2. To download a pre-configured model:
   - Browse available models
   - Click "Download" on a model of your choice (e.g., Phi-2)
3. To upload your own model:
   - Drag and drop a model file onto the upload area
   - Or click the upload area to select a file
   - Supported formats: .gguf, .bin, .pt, .pth, .model

#### Using the API

Download a pre-configured model:
```bash
curl -X POST http://localhost:5000/api/download-model/phi-2.Q4_K_M.gguf
```

Upload your own model:
```bash
curl -X POST http://localhost:5000/api/models/upload \
  -F "model_file=@/path/to/your/model.gguf"
```

### Add Documents for RAG

Add documents to the vector store for RAG:

```bash
curl -X POST http://localhost:5000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This is a sample document about artificial intelligence.", "Another document about machine learning."],
    "metadata": [{"source": "Sample 1"}, {"source": "Sample 2"}]
  }'
```

### Run a chat completion

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-2.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

### Run RAG (Retrieval-Augmented Generation)

Use the RAG endpoint to answer questions based on your documents:

```bash
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "model": "phi-2.Q4_K_M.gguf",
    "use_history": true
  }'
```

## Core Features

### Vector Document Storage

Store and retrieve documents with metadata:

```bash
# Add documents
curl -X POST http://localhost:5000/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Document content goes here"],
    "metadata": [{"source": "Book", "author": "John Doe"}]
  }'

# Search documents
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search term",
    "limit": 5,
    "filter": {"author": "John Doe"}
  }'
```

### Response History

Response history is enabled by default. All queries and responses are automatically stored for future context. 

To disable response history:

```bash
# Disable temporarily for a single session
ENABLE_RESPONSE_HISTORY=false local-ai-server start

# Or disable permanently by adding to your shell configuration:
echo 'export ENABLE_RESPONSE_HISTORY=false' >> ~/.bashrc  # For bash
echo 'export ENABLE_RESPONSE_HISTORY=false' >> ~/.zshrc   # For zsh
```

Available history endpoints:
```bash
# Search history
curl "http://localhost:5000/api/history?query=previous%20search&limit=5"

# Clean old history
curl -X POST http://localhost:5000/api/history/clean \
  -H "Content-Type: application/json" \
  -d '{"days": 30}'

# Clear all history
curl -X POST http://localhost:5000/api/history/clear

# Check history status
curl http://localhost:5000/api/history/status

# Get history for a specific user
curl "http://localhost:5000/api/history/get_for_user?user_id=your-user-id&limit=20"
```

### Retrieval-Augmented Generation (RAG)

LocalAIServer provides powerful RAG capabilities that combine document retrieval with language models:

```bash
# Basic RAG query
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "model": "phi-2.Q4_K_M.gguf"
  }'
```

#### Automatic RAG Integration

You can enable automatic RAG for all chat completions, which ensures all queries benefit from document retrieval:

```bash
# Use OpenAI-compatible endpoint with retrieval
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-2.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "use_retrieval": true,
    "search_params": {
      "limit": 5,
      "filter": {"category": "technology"}
    }
  }'
```

To enable automatic RAG by default for all chat completions:

1. Edit `endpoints.py` to change the default value:
   ```python
   use_retrieval = data.get('use_retrieval', True)  # Change from False to True
   ```

2. This seamlessly integrates both document knowledge and conversation history into every response, providing more informative and contextual answers.

The LocalAIServer codebase is already well-structured for this integration:

1. The `/v1/chat/completions` endpoint supports RAG through the `use_retrieval` parameter
2. Changing `use_retrieval` to default to `true` enables automatic RAG for all queries
3. The history management is robust with ChromaDB as the backend, ensuring reliable conversation context

### Model Management

#### Uploading Custom Models

LocalAIServer now supports uploading your own model files. This is useful for:
- Using specialized fine-tuned models
- Testing private/custom models
- Using models not included in the pre-configured list

**Supported Model Formats:**
- GGUF (`.gguf`): Recommended format, used by llama.cpp
- PyTorch models (`.pt`, `.pth`)
- Binary model files (`.bin`)
- Other model formats (`.model`)

**Upload Methods:**

1. Web Interface:
   - Go to http://localhost:5000
   - Use the drag-and-drop upload area at the top of the page
   - Wait for the upload progress to complete
   - The model will appear in the "Your Uploaded Models" section

2. API Endpoint:
   ```bash
   curl -X POST http://localhost:5000/api/models/upload \
     -F "model_file=@/path/to/your/model.gguf"
   ```

   Response:
   ```json
   {
     "status": "success",
     "message": "Model 'your-model.gguf' uploaded successfully",
     "model_id": "your-model.gguf",
     "model_path": "/path/to/models/your-model.gguf"
   }
   ```

Once uploaded, your custom models can be used like any other model:

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model.gguf",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

### OpenAI Compatibility

LocalAIServer provides drop-in replacement for OpenAI's API:

```python
import openai

# Configure to use local server
openai.api_base = "http://localhost:5000/v1"
openai.api_key = "not-needed"

# Use just like OpenAI's SDK
response = openai.ChatCompletion.create(
    model="phi-2.Q4_K_M.gguf",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print(response.choices[0].message.content)
```

## API Reference

All API endpoints are documented via Swagger UI at http://localhost:5000/docs when the server is running.

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completion |
| `/v1/completions` | POST | OpenAI-compatible text completion |
| `/v1/embeddings` | POST | OpenAI-compatible text embeddings |
| `/v1/models` | GET | List available models |
| `/v1/rag` | POST | Retrieval-Augmented Generation (OpenAI format) |
| `/api/rag` | POST | Retrieval-Augmented Generation (custom format) |
| `/api/documents` | POST | Add documents to vector store |
| `/api/documents` | DELETE | Remove documents from vector store |
| `/api/search` | POST | Search documents |
| `/api/models/upload` | POST | Upload a custom model file |
| `/api/models/all` | GET | List all installed models |
| `/api/available-models` | GET | List models available for download |
| `/api/download-model/{model_id}` | POST | Download a model |
| `/api/models/{model_id}` | DELETE | Delete a model |
| `/api/history` | GET | Search response history |
| `/api/history/clean` | POST | Clean up old history entries |
| `/api/history/clear` | POST | Delete all response history |
| `/api/history/status` | GET | Check if history is enabled |
| `/api/history/get_for_user` | GET | Get history for specific user |
| `/health` | GET | Server health check |

## Advanced RAG Features

### Document Filtering

When using RAG, you can filter documents based on metadata:

```bash
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "model": "phi-2.Q4_K_M.gguf",
    "search_params": {
      "filter": {"category": "ml", "author": "John Doe"}
    }
  }'
```

### Conversation Persistence

LocalAIServer maintains conversation history across sessions using browser cookies. This allows for follow-up questions without repeating context:

```bash
# First question
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -c cookies.txt \
  -d '{
    "model": "phi-2.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "What is a neural network?"}]
  }'

# Follow-up question (using stored cookies)
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{
    "model": "phi-2.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "How does it differ from other ML models?"}]
  }'
```

### Forced Documents for Testing

For testing RAG, you can force specific documents to be used:

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-2.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Summarize this information"}],
    "use_retrieval": true,
    "search_params": {
      "forced_documents": [
        {
          "text": "Artificial intelligence (AI) is intelligence demonstrated by machines.",
          "metadata": {"source": "Wikipedia"}
        }
      ]
    }
  }'
```

## Configuration

LocalAIServer can be configured through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_PORT` | 5000 | HTTP server port |
| `HTTPS_PORT` | 5443 | HTTPS server port |
| `VECTOR_DB_TYPE` | chroma | Vector database type (chroma or qdrant) |
| `QDRANT_PATH` | ./storage/vectors | Path for Qdrant vector storage |
| `CHROMA_PATH` | ./storage/chroma | Path for Chroma vector storage |
| `QDRANT_COLLECTION` | documents | Collection name in Qdrant |
| `CHROMA_COLLECTION` | documents | Collection name in Chroma |
| `ENABLE_RESPONSE_HISTORY` | true | Enable/disable response history |
| `MAX_HISTORY_ITEMS` | 5 | Max history items per query |
| `HISTORY_RETENTION_DAYS` | 30 | Days to retain history |

## Vector Database Options

LocalAIServer supports two vector database backends:

1. **ChromaDB** (default): A vector database with excellent concurrency handling
   - Better for multi-user deployments
   - Resilient against concurrent access issues
   - Simple architecture and good performance
   - Recommended for most use cases

2. **Qdrant**: A high-performance vector database
   - Very fast search performance
   - More advanced filtering capabilities
   - Can experience locking issues with concurrent access
   - Consider for single-user deployments with complex queries

To use Qdrant instead of ChromaDB:

```bash
# Set environment variable
export VECTOR_DB_TYPE=qdrant
local-ai-server start

# Or specify on the command line
local-ai-server start --vector-db qdrant
```

## Advanced Usage

### Running with Docker

```bash
docker run -p 5000:5000 -p 5443:5443 -v ./models:/app/models -v ./storage:/app/storage hannesnortje/local-ai-server
```

### Using Custom Models

To use your own models, place the model files in the `models` directory:

```bash
cp path/to/your/model.gguf ~/.local/share/local_ai_server/models/
```

### Integration with LangChain

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="phi-2.Q4_K_M.gguf",
    openai_api_base="http://localhost:5000/v1",
    openai_api_key="not-needed"
)

messages = [HumanMessage(content="Hello, how are you?")]
response = chat(messages)
print(response.content)
```

## Development

### Setting Up Development Environment

```bash
git clone https://github.com/hannesnortje/LocalAIServer.git
cd LocalAIServer
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Running Tests

```bash
python -m pytest
```

### Project Structure

- `local_ai_server/`: Main package
  - `__main__.py`: Entry point
  - `server.py`: Flask server setup
  - `endpoints.py`: API routes
  - `model_manager.py`: Model loading and inference
  - `vector_store.py`: Qdrant vector database
  - `chroma_store.py`: ChromaDB vector database implementation
  - `vector_store_factory.py`: Factory for selecting vector database
  - `rag.py`: Retrieval-Augmented Generation
  - `history_manager.py`: Response history
  - `app_state.py`: Global application state

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Use different ports: `HTTP_PORT=8000 HTTPS_PORT=8443 local-ai-server start`

2. **Memory issues with large models**
   - Use smaller quantized models (e.g., Q4_K_M variants)
   - Increase system swap space
   - Add CUDA support for GPU acceleration

3. **SSL Certificate Warnings**
   - The server generates self-signed certificates
   - Add exception in your browser or use HTTP for local development

4. **Vector database locks**
   - Use ChromaDB instead of Qdrant for better concurrency handling
   - Set `VECTOR_DB_TYPE=chroma` in your environment

### Logs

Logs are stored in `local_ai_server.log` in the current directory.

## License

MIT
