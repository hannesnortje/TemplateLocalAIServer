from flask import jsonify, request, Response, stream_with_context
import json
import logging
import requests
import os
import time
import uuid
import flask
from typing import List, Dict, Optional, Union, Any

from . import __version__
from .models_config import AVAILABLE_MODELS, EMBEDDING_MODEL
from .model_manager import model_manager
from .config import MODELS_DIR, ENABLE_RESPONSE_HISTORY
from .vector_store import get_vector_store
from .rag import RAG
from .history_manager import get_response_history
from .app_state import vector_store as app_vector_store, history_manager as app_history_manager

logger = logging.getLogger(__name__)

# List of valid parameters that can be passed to the model
VALID_MODEL_PARAMS = {
    'temperature', 'max_tokens', 'stream', 'top_p', 
    'frequency_penalty', 'presence_penalty', 'stop'
}

def get_persistent_user_id():
    """Get a persistent user ID that works across sessions."""
    user_id = None
    
    # Try to get from cookie first (most persistent)
    cookie_name = "localai_user_id"
    user_id = request.cookies.get(cookie_name)
    
    # If no cookie, check session (if available)
    if not user_id and hasattr(flask, 'session'):
        try:
            user_id = flask.session.get('user_id')
        except:
            pass
    
    # If still no user ID, generate new ID
    if not user_id:
        user_id = str(uuid.uuid4())
        
    return user_id

def setup_routes(app):
    @app.route("/api/available-models", methods=['GET'])
    def get_available_models():
        """Get list of available models for download"""
        return jsonify(AVAILABLE_MODELS)

    @app.route("/api/models/all", methods=['GET'])
    def list_all_models():
        """List all installed models"""
        models = model_manager.get_status()
        return jsonify([{
            "name": name,
            "type": info.model_type or "unknown",
            "loaded": info.loaded,
            "context_size": info.context_window,
            "description": info.description if hasattr(info, 'description') else None,
            "custom_upload": name not in AVAILABLE_MODELS  # Flag to identify custom uploads
        } for name, info in models.items()])

    @app.route("/v1/models", methods=['GET'])
    def list_models():
        """List installed models"""
        return jsonify({"data": model_manager.list_models()})

    @app.route("/v1/chat/completions", methods=['POST'])
    def chat_completion():
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            model_name = data.get('model')
            if not model_name:
                return jsonify({"error": "Model name is required"}), 400

            messages = data.get('messages', [])
            if not messages:
                return jsonify({"error": "Messages array is required"}), 400

            # Extract only valid parameters that are explicitly provided
            params = {k: v for k, v in data.items() if k in VALID_MODEL_PARAMS and v is not None}
            
            # Get a persistent user ID that works across chat resets
            user_id = get_persistent_user_id()
            logger.debug(f"User ID for request: {user_id}")
            
            # Create a response object to modify later
            response_obj = None
            
            # Check if RAG should be used
            use_retrieval = data.get('use_retrieval', False)
            
            if use_retrieval:
                # Get the last message as the query
                if messages[-1]['role'] != 'user':
                    return jsonify({"error": "Last message must be from the user when using retrieval"}), 400
                
                query = messages[-1]['content']
                
                # Extract search parameters - add debug logging
                search_params = data.get('search_params', {})
                logger.debug(f"Original search params: {search_params}")
                
                # Add user_id to search params for history filtering
                if 'search_params' not in data:
                    data['search_params'] = {}
                data['search_params']['user_id'] = user_id
                
                # Allow direct document injection for testing (pass through if provided)
                if 'forced_documents' in search_params:
                    logger.debug(f"Forced documents provided: {len(search_params['forced_documents'])}")
                
                # Debug log the full search params
                logger.debug(f"RAG search params: {data['search_params']}")
                
                # Handle streaming for RAG
                if params.get('stream', False):
                    def generate():
                        try:
                            # Generate RAG response
                            rag_params = params.copy()
                            rag_params.pop('stream', None)
                            
                            response = RAG.generate_rag_response(
                                query=query,
                                model_name=model_name,
                                search_params=search_params,
                                generation_params=rag_params
                            )
                            
                            doc_count = len(response['retrieved_documents'])
                            rag_answer = response['answer']
                            
                            yield f"data: {json.dumps({
                                'id': f'chatrag_{int(time.time())}',
                                'object': 'chat.completion.chunk',
                                'created': int(time.time()),
                                'model': model_name,
                                'choices': [{
                                    'index': 0,
                                    'delta': {
                                        'role': 'assistant',
                                        'content': rag_answer
                                    },
                                    'finish_reason': 'stop'
                                }],
                                'usage': {
                                    'prompt_tokens': 0,  # Estimated
                                    'completion_tokens': len(rag_answer.split()),
                                    'total_tokens': len(rag_answer.split())
                                }
                            })}\n\n"
                            yield "data: [DONE]\n\n"
                        except Exception as e:
                            logger.error(f"RAG streaming error: {e}")
                            yield f"data: {json.dumps({'error': str(e)})}\n\n"

                    return Response(
                        stream_with_context(generate()), 
                        mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
                    )
                
                # Regular RAG response
                response = RAG.generate_rag_response(
                    query=query,
                    model_name=model_name,
                    search_params=data['search_params'],  # Use updated search params with user_id
                    generation_params=params
                )
                
                # When saving to history, include user ID
                if app_history_manager is not None:
                    try:
                        app_history_manager.save_response(
                            query=query,
                            response=response["answer"],
                            metadata={
                                "timestamp": time.time(),
                                "model": model_name,
                                "document_count": len(response.get("retrieved_documents", [])),
                                "user_id": user_id  # Store user ID in metadata
                            }
                        )
                    except Exception as history_save_error:
                        logger.warning(f"Failed to save to history: {history_save_error}")
                
                # Include debugging info in response for testing
                debug_info = {}
                if app.config.get('TESTING', False):
                    debug_info = {
                        "debug": {
                            "user_id": user_id,
                            "retrieved_doc_count": len(response.get("retrieved_documents", [])),
                            "history_count": len(response.get("history_items", [])),
                            "prompt": response.get("metadata", {}).get("rag_prompt", "")
                        }
                    }
                
                # Create final response with user_id cookie
                response_obj = jsonify({
                    "id": f"chatrag_{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response["answer"]
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": sum(len(m.get('content', '').split()) for m in messages),
                        "completion_tokens": len(response["answer"].split()),
                        "total_tokens": sum(len(m.get('content', '').split()) for m in messages) + len(response["answer"].split())
                    },
                    **debug_info
                })
                
            else:
                # Regular chat completion (non-RAG) - continue with existing code
                # Load model if needed
                if model_manager.model is None or model_manager.current_model_name != model_name:
                    try:
                        model_manager.load_model(model_name)
                    except Exception as e:
                        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

                # Handle streaming response
                if params.get('stream', False):
                    def generate():
                        try:
                            response = model_manager.create_chat_completion(messages, **params)
                            yield f"data: {json.dumps({
                                'id': f'chat_{int(time.time())}',
                                'object': 'chat.completion.chunk',
                                'created': int(time.time()),
                                'model': model_name,
                                'choices': [{
                                    'index': 0,
                                    'delta': response,
                                    'finish_reason': 'stop'
                                }]
                            })}\n\n"
                            yield "data: [DONE]\n\n"
                        except Exception as e:
                            logger.error(f"Streaming error: {e}")
                            yield f"data: {json.dumps({'error': str(e)})}\n\n"

                    return Response(
                        stream_with_context(generate()), 
                        mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
                    )

                # Get ALL previous conversations for this user
                history_items = []
                if ENABLE_RESPONSE_HISTORY and app_history_manager:
                    try:
                        # Get all history for this user without semantic filtering
                        history_items = app_history_manager.get_user_history(user_id, limit=20)
                        
                        # Sort by timestamp if available
                        history_items.sort(
                            key=lambda x: x.get("metadata", {}).get("timestamp", 0)
                        )
                        
                        logger.debug(f"Retrieved {len(history_items)} previous conversations for user {user_id}")
                    except Exception as e:
                        logger.warning(f"Error getting user history: {e}")
                
                # Format history as messages
                history_messages = []
                for item in history_items:
                    history_messages.append({"role": "user", "content": item["query"]})
                    history_messages.append({"role": "assistant", "content": item["response"]})
                
                # Combine with current messages
                all_messages = history_messages + messages
                
                # Trim if needed for model context window
                if len(all_messages) > 20:  # Adjust based on model context size
                    all_messages = all_messages[-20:]  # Keep only most recent conversations
                
                # Generate response
                response = model_manager.create_chat_completion(all_messages, **params)
                
                # Store in history with user_id
                if ENABLE_RESPONSE_HISTORY and app_history_manager:
                    try:
                        # Get most recent user message
                        user_messages = [msg for msg in messages if msg.get('role') == 'user']
                        current_query = user_messages[-1]['content'] if user_messages else ""
                        
                        app_history_manager.save_response(
                            query=current_query,
                            response=response['content'],
                            metadata={
                                "timestamp": time.time(),
                                "model": model_name,
                                "user_id": user_id  # Store user ID in metadata
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Error saving to history: {e}")
                
                # Create response with cookie
                response_obj = jsonify({
                    "id": f"chat_{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": response,
                        "finish_reason": 'stop'
                    }]
                })
            
            # Set user_id cookie for 1 year and make it available in all paths
            response_obj.set_cookie(
                'localai_user_id',
                user_id,
                max_age=31536000,  # 1 year in seconds
                httponly=True,
                samesite='Lax',
                path='/'  # Make cookie available across all paths
            )
            
            # Store user_id in session as well if available
            if hasattr(flask, 'session'):
                try:
                    flask.session['user_id'] = user_id
                    flask.session.modified = True
                except:
                    pass
            
            return response_obj

        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/health")
    def health_check():
        """Check server health status"""
        return jsonify({
            "status": "healthy",
            "version": __version__
        })

    @app.route("/api/download-model/<model_id>", methods=['POST'])
    def download_model(model_id: str):
        """Download a model with progress streaming"""
        if model_id not in AVAILABLE_MODELS:
            return jsonify({"error": "Model not found"}), 404
        
        model_info = AVAILABLE_MODELS[model_id]
        target_path = MODELS_DIR / model_id
        temp_path = target_path.with_suffix('.tmp')

        def download_stream():
            if target_path.exists():
                yield json.dumps({
                    "status": "exists",
                    "progress": 100,
                    "message": "Model already downloaded"
                }) + "\n"
                return

            try:
                response = requests.get(model_info["url"], stream=True)
                if response.status_code != 200:
                    if temp_path.exists():
                        temp_path.unlink()
                    yield json.dumps({
                        "status": "error",
                        "message": "Download failed"
                    }) + "\n"
                    return

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = int((downloaded / total_size) * 100) if total_size else 0
                            
                            yield json.dumps({
                                "status": "downloading",
                                "progress": progress,
                                "downloaded": downloaded,
                                "total": total_size
                            }) + "\n"

                temp_path.rename(target_path)
                yield json.dumps({
                    "status": "success",
                    "progress": 100,
                    "message": "Model downloaded successfully",
                    "model_id": model_id
                }) + "\n"

            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                logger.error(f"Download error: {e}")
                yield json.dumps({
                    "status": "error",
                    "message": str(e)
                }) + "\n"

        return Response(stream_with_context(download_stream()), 
                      mimetype='application/x-ndjson')

    @app.route("/api/models/<model_id>", methods=['DELETE'])
    def delete_model(model_id: str):
        """Delete a model"""
        model_path = MODELS_DIR / model_id
        if not model_path.exists():
            return jsonify({"error": "Model not found"}), 404
        
        try:
            model_path.unlink()
            return jsonify({"status": "success", "message": f"Model {model_id} deleted successfully"})
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return jsonify({"error": "Failed to delete model"}), 500

    @app.route("/api/documents", methods=['POST'])
    def add_documents():
        """Add documents to the vector store"""
        try:
            vector_store = get_vector_store()
            data = request.get_json()
            if not data or 'texts' not in data:
                return jsonify({"error": "Missing texts in request"}), 400

            texts = data['texts']
            metadata = data.get('metadata', [{}] * len(texts))

            if len(metadata) != len(texts):
                return jsonify({"error": "Metadata length must match texts length"}), 400

            ids = vector_store.add_texts(texts, metadata)
            return jsonify({
                "status": "success",
                "ids": ids,
                "count": len(ids)
            })
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/search", methods=['POST'])
    def search_documents():
        """Search for similar documents"""
        try:
            vector_store = get_vector_store()
            data = request.get_json()
            if not data or 'query' not in data:
                return jsonify({"error": "Missing query in request"}), 400

            query = data['query']
            k = data.get('limit', 4)
            filter_params = data.get('filter')

            results = vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_params
            )
            return jsonify({
                "status": "success",
                "results": results
            })
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/documents", methods=['DELETE'])
    def delete_documents():
        """Delete documents from the vector store"""
        try:
            vector_store = get_vector_store()
            data = request.get_json()
            if not data or 'ids' not in data:
                return jsonify({"error": "Missing ids in request"}), 400

            vector_store.delete_texts(data['ids'])
            return jsonify({
                "status": "success",
                "message": f"Deleted {len(data['ids'])} documents"
            })
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/v1/embeddings", methods=['POST'])
    def create_embeddings():
        """Create embeddings using the OpenAI-compatible format"""
        try:
            vector_store = get_vector_store()
            data = request.get_json()
            if not data or 'input' not in data:
                return jsonify({"error": "Missing input in request"}), 400

            input_texts = data['input']
            if isinstance(input_texts, str):
                input_texts = [input_texts]

            embeddings = vector_store.model.encode(input_texts, convert_to_numpy=True)
            
            return jsonify({
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": embedding.tolist(),
                        "index": i
                    } for i, embedding in enumerate(embeddings)
                ],
                "model": EMBEDDING_MODEL,
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in input_texts),
                    "total_tokens": sum(len(text.split()) for text in input_texts)
                }
            })
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/v1/completions", methods=['POST'])
    def create_completion():
        """Create completion using the OpenAI-compatible format"""
        try:
            data = request.get_json()
            if not data or 'prompt' not in data:
                return jsonify({"error": "Missing prompt in request"}), 400

            model_name = data.get('model')
            if not model_name:
                return jsonify({"error": "Model name is required"}), 400

            # Extract parameters
            params = {k: v for k, v in data.items() if k in VALID_MODEL_PARAMS and v is not None}
            
            # Load model if needed
            if model_manager.model is None or model_manager.current_model_name != model_name:
                model_manager.load_model(model_name)

            # Handle streaming
            if params.get('stream', False):
                def generate():
                    response = model_manager.generate(data['prompt'], **params)
                    chunk = {
                        "id": f"cmpl-{int(time.time())}",
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "text": response,
                            "index": 0,
                            "finish_reason": "stop",
                            "logprobs": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return Response(
                    stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache'}
                )

            # Handle regular response
            response = model_manager.generate(data['prompt'], **params)
            return jsonify({
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "text": response,
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None
                }],
                "usage": {
                    "prompt_tokens": len(data['prompt'].split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(data['prompt'].split()) + len(response.split())
                }
            })

        except Exception as e:
            logger.error(f"Error in completion: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/v1/rag", methods=['POST'])
    def rag_completion():
        """Generate response using RAG (Retrieval-Augmented Generation)"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            query = data.get('query')
            if not query:
                return jsonify({"error": "Query is required"}), 400
                
            model_name = data.get('model')
            if not model_name:
                return jsonify({"error": "Model name is required"}), 400
            
            # Get history preference (default to global setting)
            use_history = data.get('use_history', ENABLE_RESPONSE_HISTORY)
                
            # Extract parameters for search and generation
            search_params = data.get('search_params', {})
            generation_params = {k: v for k, v in data.items() if k in VALID_MODEL_PARAMS and v is not None}
            
            # Handle streaming
            if generation_params.get('stream', False):
                def generate():
                    try:
                        # Generate response using RAG
                        generation_params_copy = generation_params.copy()
                        generation_params_copy.pop('stream', None)
                        response = RAG.generate_rag_response(
                            query=query,
                            model_name=model_name,
                            search_params=search_params,
                            generation_params=generation_params_copy,
                            use_history=use_history
                        )
                        
                        yield f"data: {json.dumps({
                            'id': f'rag_{int(time.time())}',
                            'object': 'rag.completion.chunk',
                            'created': int(time.time()),
                            'model': model_name,
                            'answer': response['answer'],
                            'retrieved_document_count': len(response['retrieved_documents']),
                            'history_count': len(response.get('history_items', [])),
                            'finish_reason': 'stop'
                        })}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logger.error(f"RAG streaming error: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
                return Response(
                    stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
                )
            
            # Generate response using RAG
            response = RAG.generate_rag_response(
                query=query,
                model_name=model_name,
                search_params=search_params,
                generation_params=generation_params,
                use_history=use_history
            )
            
            return jsonify({
                "id": f"rag_{int(time.time())}",
                "object": "rag.completion",
                "created": int(time.time()),
                "model": model_name,
                "answer": response["answer"],
                "retrieved_documents": response["retrieved_documents"],
                "history_items": response.get("history_items", []),
                "metadata": response["metadata"]
            })
            
        except Exception as e:
            logger.error(f"Error in RAG completion: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Add history management endpoints
    @app.route("/api/history", methods=['GET'])
    def search_history():
        """Search response history"""
        try:
            if not ENABLE_RESPONSE_HISTORY:
                return jsonify({"error": "Response history is disabled"}), 400
                
            history_manager = get_response_history()  # Use factory function
            query = request.args.get('query', '')
            limit = int(request.args.get('limit', 10))
            min_score = float(request.args.get('min_score', 0.7))
            
            # Parse filter parameters (format: filter.key=value)
            filter_params = {}
            for key, value in request.args.items():
                if (key.startswith('filter.')):
                    filter_key = key[7:]  # Remove 'filter.' prefix
                    filter_params[filter_key] = value
            
            results = history_manager.find_similar_responses(
                query=query,
                limit=limit,
                min_score=min_score,
                filter_params=filter_params if filter_params else None
            )
            
            return jsonify({
                "status": "success",
                "results": results,
                "count": len(results)
            })
            
        except Exception as e:
            logger.error(f"Error searching history: {e}")
            return jsonify({"error": str(e)}), 500
            
    @app.route("/api/history/clean", methods=['POST'])
    def clean_history():
        """Clean old history entries"""
        try:
            if not ENABLE_RESPONSE_HISTORY:
                return jsonify({"error": "Response history is disabled"}), 400
                
            history_manager = get_response_history()  # Use factory function
            data = request.get_json() or {}
            days = int(data.get('days', 30))
            
            count = history_manager.clean_old_entries(days=days)
            
            return jsonify({
                "status": "success",
                "message": f"Cleaned {count} old history entries",
                "count": count
            })
            
        except Exception as e:
            logger.error(f"Error cleaning history: {e}")
            return jsonify({"error": str(e)}), 500
            
    @app.route("/api/history/clear", methods=['POST'])
    def clear_history():
        """Clear all history"""
        try:
            if not ENABLE_RESPONSE_HISTORY:
                return jsonify({"error": "Response history is disabled"}), 400
                
            history_manager = get_response_history()  # Use factory function
            success = history_manager.delete_all_history()
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": "All history cleared"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to clear history"
                }), 500
                
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return jsonify({"error": str(e)}), 500
            
    @app.route("/api/history/status", methods=['GET'])
    def history_status():
        """Get history status"""
        return jsonify({
            "enabled": ENABLE_RESPONSE_HISTORY
        })

    @app.route("/api/history/get_for_user", methods=['GET'])
    def get_user_history():
        """Get all history for a specific user"""
        try:
            if not ENABLE_RESPONSE_HISTORY:
                return jsonify({"error": "Response history is disabled"}), 400
                
            user_id = request.args.get('user_id')
            if not user_id:
                user_id = get_persistent_user_id()
                
            limit = int(request.args.get('limit', 20))
            
            history_manager = get_response_history()
            results = history_manager.get_user_history(user_id, limit)
            
            return jsonify({
                "status": "success",
                "user_id": user_id,
                "results": results,
                "count": len(results)
            })
            
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/models/upload", methods=['POST'])
    def upload_model():
        """Upload a model file"""
        try:
            if 'model_file' not in request.files:
                return jsonify({"error": "No file provided"}), 400

            model_file = request.files['model_file']
            
            # Check if filename is empty
            if model_file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Check file extension
            allowed_extensions = {'.gguf', '.bin', '.pt', '.pth', '.model'}
            filename = model_file.filename
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                return jsonify({
                    "error": f"Unsupported file type: {file_ext}",
                    "allowed_extensions": list(allowed_extensions)
                }), 400
            
            # Create models directory if it doesn't exist
            os.makedirs(MODELS_DIR, exist_ok=True)
            
            # Save the file
            file_path = os.path.join(MODELS_DIR, filename)
            model_file.save(file_path)
            
            # Check if file was saved successfully
            if os.path.exists(file_path):
                # For GGUF files, try loading some model info
                model_info = {}
                model_type = "unknown"
                context_window = None
                
                if file_ext == '.gguf':
                    try:
                        from llama_cpp import Llama
                        # Just peek at the model metadata without fully loading it
                        model = Llama(model_path=file_path, n_ctx=8, verbose=False, n_threads=1)
                        context_window = model.n_ctx()
                        model_type = "gguf"
                        model_info = {
                            "context_window": context_window,
                            "model_type": model_type
                        }
                    except Exception as model_error:
                        logger.warning(f"Could not extract info from model: {model_error}")
                
                # Update model manager's status cache to include this model immediately
                # This ensures it appears in the UI right away
                try:
                    model_manager.update_model_info(
                        model_name=filename,
                        model_type=model_type,
                        context_window=context_window
                    )
                except Exception as update_error:
                    logger.warning(f"Could not update model cache: {update_error}")
                
                return jsonify({
                    "status": "success",
                    "message": f"Model '{filename}' uploaded successfully",
                    "model_id": filename,
                    "model_path": str(file_path),
                    "model_info": model_info
                })
            else:
                return jsonify({"error": "Failed to save file"}), 500
            
        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/rag", methods=['POST'])
    def api_rag():
        """Generate a response using RAG for direct API access (non-OpenAI-compatible endpoint)"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            query = data.get('query')
            if not query:
                return jsonify({"error": "Query is required"}), 400
                
            model_name = data.get('model')
            if not model_name:
                return jsonify({"error": "Model name is required"}), 400
            
            # Extract search and generation parameters
            search_params = data.get('search_params', {})
            use_history = data.get('use_history', ENABLE_RESPONSE_HISTORY)
            
            # Extract generation parameters that are valid model parameters
            generation_params = {k: v for k, v in data.items() if k in VALID_MODEL_PARAMS and v is not None}
            
            # Get user ID for history
            user_id = get_persistent_user_id()
            if user_id and 'user_id' not in search_params:
                search_params['user_id'] = user_id
            
            # Generate RAG response
            response = RAG.generate_rag_response(
                query=query,
                model_name=model_name,
                search_params=search_params,
                generation_params=generation_params,
                use_history=use_history
            )
            
            result = {
                "answer": response["answer"],
                "model": model_name,
                "retrieved_documents": response["retrieved_documents"],
                "metadata": response["metadata"]
            }
            
            if use_history:
                result["history_items"] = response.get("history_items", [])
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error in RAG endpoint: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/scrape", methods=['GET'])
    def scrape_web():
        """Scrape web content based on a search query and optionally save to the vector store."""
        try:
            # Get query parameters
            query = request.args.get('query')
            if not query:
                return jsonify({"error": "Query parameter is required"}), 400
                
            # EU Compliance Parameters
            explicit_consent = request.args.get('explicit_consent', 'false').lower() == 'true'
            privacy_policy_url = request.args.get('privacy_policy_url', '')
            consent_timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            
            # Check for explicit consent (EU compliance)
            if not explicit_consent:
                return jsonify({
                    "error": "Explicit consent is required for web scraping under EU regulations",
                    "message": "Please provide 'explicit_consent=true' and review the privacy policy",
                    "consent_required": True,
                    "privacy_policy_url": privacy_policy_url or "Please provide a privacy policy URL"
                }), 403
                
            # Optional parameters
            engine = request.args.get('engine', 'google').lower()
            num_results = int(request.args.get('num_results', 5))
            fetch_content = request.args.get('fetch_content', 'false').lower() == 'true'
            timeout = int(request.args.get('timeout', 30))
            headless = request.args.get('headless', 'true').lower() == 'true'
            
            # New parameter to determine whether to save to vector store
            save_to_db = request.args.get('save_to_db', 'false').lower() == 'true'
            
            try:
                # Import here to avoid dependency issues if not installed
                from .web_scraper import search_and_scrape
            except ImportError:
                return jsonify({
                    "error": "Web scraping dependencies are not installed. Please install with: \n\n"
                             "If using pipx: pipx inject localaiserver selenium webdriver-manager\n\n"
                             "Or with pip: pip install selenium webdriver-manager"
                }), 500
            
            logger.info(f"Starting web scrape for query: '{query}' (engine: {engine}, save_to_db: {save_to_db}, consent: {explicit_consent})")
            
            # Perform the search and scrape
            results = search_and_scrape(
                query=query,
                engine=engine,
                num_results=num_results,
                fetch_content=fetch_content,
                timeout=timeout,
                headless=headless
            )
            
            # Add consent information to results
            results['explicit_consent'] = explicit_consent
            results['consent_timestamp'] = consent_timestamp
            results['privacy_policy_url'] = privacy_policy_url
            
            # If requested, save results to vector store
            if save_to_db and results['results']:
                try:
                    vector_store = get_vector_store()
                    
                    # Prepare documents for vector store
                    texts = []
                    metadata_list = []
                    
                    for result in results['results']:
                        # If we have content, use that, otherwise use the snippet
                        if fetch_content and 'content' in result:
                            content = result['content']
                        else:
                            content = result.get('snippet', '')
                        
                        if not content:
                            continue
                            
                        # Include the title in the content for better context
                        document = f"{result['title']}\n\n{content}"
                        
                        # Create metadata entry
                        meta = {
                            "url": result.get('url', ''),
                            "title": result.get('title', ''),
                            "source": "web_scrape",
                            "query": query,
                            "scrape_timestamp": results['timestamp'],
                            "search_engine": engine,
                            # Add consent information to metadata
                            "explicit_consent": explicit_consent,
                            "consent_timestamp": consent_timestamp,
                            "privacy_policy_url": privacy_policy_url
                        }
                        
                        texts.append(document)
                        metadata_list.append(meta)
                    
                    if texts:
                        # Add to vector store
                        ids = vector_store.add_texts(texts, metadata_list)
                        logger.info(f"Added {len(ids)} documents to vector store from web scrape")
                        
                        # Add IDs to the results for reference
                        results['saved_document_ids'] = ids
                        results['saved_to_db'] = True
                except Exception as e:
                    logger.error(f"Error saving web scrape results to vector store: {e}")
                    results['save_error'] = str(e)
                    results['saved_to_db'] = False
            else:
                results['saved_to_db'] = False
            
            return jsonify(results)
            
        except Exception as e:
            logger.error(f"Web scraping error: {str(e)}", exc_info=True)
            return jsonify({"error": f"Web scraping failed: {str(e)}"}), 500
