import logging
from typing import List, Dict, Optional, Union, Any
import time
import numpy as np
from scipy.spatial.distance import cosine

from .vector_store_factory import get_vector_store
from .model_manager import model_manager
from .history_manager import get_response_history  # Use the factory function instead
from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, ENABLE_RESPONSE_HISTORY

# Import the global vector_store and history_manager from server.py
from .app_state import vector_store as app_vector_store, history_manager as app_history_manager

logger = logging.getLogger(__name__)

class RAG:
    """Retrieval-Augmented Generation utility."""
    
    # We'll use the application-level shared instances by default
    vector_store = None  # Will fall back to app_vector_store if None
    
    @staticmethod
    def format_retrieved_documents(docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for inclusion in the prompt.
        
        Args:
            docs: List of document dictionaries with 'text' and 'metadata' keys
            
        Returns:
            Formatted string with document contents
        """
        if not docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            source = ""
            if doc.get("metadata") and doc["metadata"].get("source"):
                source = f" (Source: {doc['metadata']['source']})"
                
            formatted_docs.append(f"Document {i}{source}:\n{doc['text']}\n")
            
        return "\n".join(formatted_docs)
    
    @staticmethod
    def format_history_responses(history_items: List[Dict]) -> str:
        """Format historical responses for inclusion in the prompt.
        
        Args:
            history_items: List of historical response dictionaries
            
        Returns:
            Formatted string with historical responses
        """
        if not history_items:
            return ""
        
        history_text = ["PREVIOUS QUESTIONS AND ANSWERS:"]
        
        for i, item in enumerate(history_items, 1):
            timestamp = item.get("metadata", {}).get("timestamp", "")
            if timestamp:
                date_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))
                history_text.append(f"Date: {date_str}")
                
            history_text.append(f"User: {item['query']}")
            history_text.append(f"Assistant: {item['response']}")
            history_text.append("")  # Empty line between entries
            
        return "\n".join(history_text)
    
    @staticmethod
    def generate_rag_response(
        query: str,
        model_name: str,
        search_params: Optional[Dict] = None,
        generation_params: Optional[Dict] = None,
        use_history: Optional[bool] = None
    ) -> Dict:
        """Generate a response using retrieved documents as context."""
        start_time = time.time()
        
        # Set default parameters
        search_params = search_params or {}
        generation_params = generation_params or {}
        
        # Determine if we should use history
        if use_history is None:
            use_history = ENABLE_RESPONSE_HISTORY
        
        # Use global instances from the application
        try:
            # Use class vector store if set, otherwise use application instance
            vector_store = RAG.vector_store or app_vector_store
            
            # Set search parameters
            k = search_params.get('limit', 4)
            filter_params = search_params.get('filter')
            
            # Get user ID for history lookup
            user_id = search_params.get('user_id', '')
            logger.debug(f"RAG using user_id: {user_id}")
            
            # Get forced documents if provided (useful for testing)
            forced_documents = search_params.get('forced_documents', [])
            
            # If we have forced documents, use those directly
            if forced_documents:
                logger.debug(f"Using {len(forced_documents)} forced documents for RAG")
                results = forced_documents
            else:
                # Retrieve documents first
                # Check if vector store exists
                if vector_store is None:
                    logger.warning("No vector store available. Proceeding without document retrieval.")
                    results = []
                else:
                    # Use direct query for search - don't enhance it yet
                    logger.debug(f"RAG initial search query: '{query}'")
                    
                    # Set higher k to get more potential matches
                    initial_k = min(k * 3, 12)  # Get more results initially, then filter
                    
                    # Retrieve relevant documents using direct query
                    results = vector_store.similarity_search(
                        query=query,
                        k=initial_k,
                        filter=filter_params
                    )
                    
                    # Log retrieved documents for debugging
                    if results:
                        logger.debug(f"Retrieved {len(results)} documents")
                        for i, r in enumerate(results[:3]):  # Log first 3 docs
                            logger.debug(f"Doc {i}: {r.get('text', '')[:100]}...")
                    else:
                        logger.debug("No documents retrieved")
            
            # If query contains specific terms that should match documents exactly,
            # do a direct text search as a fallback if vector search failed
            if not results and vector_store:
                # Extract important terms from query (words in quotes, proper nouns, etc.)
                import re
                # Look for quoted terms or words with underscores or specific patterns
                special_terms = re.findall(r'"([^"]+)"|\'([^\']+)\'|(\w+_\w+)', query)
                terms = []
                for term_group in special_terms:
                    term = next((t for t in term_group if t), None)
                    if term:
                        terms.append(term)
                
                if terms:
                    logger.debug(f"Doing direct text search for terms: {terms}")
                    for term in terms:
                        # Try direct search for each term
                        direct_results = vector_store.similarity_search(
                            query=term,
                            k=k,
                            filter=filter_params
                        )
                        
                        if direct_results:
                            logger.debug(f"Found {len(direct_results)} documents with direct search for '{term}'")
                            results = direct_results
                            break
            
            # Get historical responses if enabled - AFTER document retrieval
            history_items = []
            if use_history and app_history_manager is not None and user_id:  # IMPORTANT: only get history if user_id exists
                try:
                    history_manager = app_history_manager  # Use application instance
                    
                    # First get direct user history without semantic search
                    # This ensures we get the actual conversation history for this user
                    direct_history = history_manager.get_user_history(
                        user_id=user_id,
                        limit=10  # Get more direct history
                    )
                    
                    if direct_history:
                        logger.debug(f"Found {len(direct_history)} direct history items for user {user_id}")
                        history_items = direct_history
                    else:
                        # Fallback to semantic search with user filter only if direct history fails
                        logger.debug(f"No direct history for user {user_id}, trying semantic search")
                        history_limit = search_params.get('history_limit', 5)
                        
                        # Create user filter
                        user_filter = {"user_id": user_id}
                        
                        # Get history with user filter and very low similarity score to catch all items
                        history_items = history_manager.find_similar_responses(
                            query=query,
                            limit=history_limit,
                            filter_params=user_filter,
                            min_score=0.1  # Very low threshold to ensure we get user's history
                        )
                    
                    # Sort history by timestamp to ensure correct order
                    if history_items:
                        history_items.sort(
                            key=lambda x: x.get("metadata", {}).get("timestamp", 0)
                        )
                        
                        # Log history items for debugging
                        logger.debug(f"Using {len(history_items)} history items for user {user_id}")
                        for i, item in enumerate(history_items):
                            logger.debug(f"History item {i}: Q={item['query'][:30]}... A={item['response'][:30]}...")
                except Exception as history_error:
                    logger.warning(f"Failed to fetch history: {history_error}")
                    # Continue without history
            
            # Now try again with enhanced query if we didn't get good results
            if not results and history_items:
                # If initial search found nothing but we have history,
                # try a second search with enhanced query
                enhanced_query = query
                recent_history = sorted(
                    history_items, 
                    key=lambda x: x.get("metadata", {}).get("timestamp", 0),
                    reverse=True  # Most recent first
                )[:2]  # Only use latest two conversations
                
                if recent_history:
                    # Combine most recent query with current query
                    recent_query = recent_history[0]['query']
                    enhanced_query = f"{recent_query} {query}"
                    logger.debug(f"Enhanced search query: {enhanced_query[:100]}...")
                    
                    # Try search again with enhanced query
                    results = vector_store.similarity_search(
                        query=enhanced_query,
                        k=k,
                        filter=filter_params
                    )
                    
                    if results:
                        logger.debug(f"Second attempt retrieved {len(results)} documents")
            
            # Score the results to find most relevant docs for this query
            if len(results) > k:
                # If we have more results than needed, filter them to the most relevant
                query_embedding = vector_store.model.encode(query, convert_to_numpy=True)
                
                # Re-score documents based on direct comparison to the query
                for doc in results:
                    doc_text = doc['text']
                    doc_embedding = vector_store.model.encode(doc_text, convert_to_numpy=True)
                    
                    # Replace the util.cos_sim call with direct cosine similarity calculation
                    # Lower value means more similar (0 is identical, 1 is completely different)
                    similarity = cosine(query_embedding, doc_embedding)
                    doc['direct_score'] = similarity
                
                # Sort by direct similarity and take top k (lower score is better)
                results = sorted(results, key=lambda x: x.get('direct_score', 1.0))[:k]
            
            # Format retrieved documents and history for the prompt
            retrieved_docs = RAG.format_retrieved_documents(results)
            history_context = RAG.format_history_responses(history_items)
            
            logger.debug(f"Using {len(results)} documents and {len(history_items)} history items")
            
            # Load the specified model
            if model_manager.model is None or model_manager.current_model_name != model_name:
                model_manager.load_model(model_name)
            
            # Determine whether to use RAG or standard LLM mode based on document retrieval
            # Check if we actually found relevant documents
            using_rag_mode = len(results) > 0
            
            # Create appropriate system instruction based on available context
            if using_rag_mode:
                system_instruction = "You are an AI assistant that answers questions based on the provided DOCUMENTS and conversation history."
                
                # Use the RAG-specific prompt that emphasizes document information
                rag_prompt = f"""{system_instruction}

IMPORTANT: You must ONLY use information from these DOCUMENTS to answer the question.

DOCUMENTS:
{retrieved_docs}

PREVIOUS CONVERSATION:
{history_context}

USER QUESTION: {query}

CRITICAL INSTRUCTIONS:
1. If the DOCUMENTS section contains information to answer the question, quote EXACT phrases, terms, and names from the documents.
2. Never paraphrase, replace, or alter specific terms and names from the documents. Use VERBATIM any proper nouns, names, or technical terms.
3. If the documents mention a system by a specific name like "ai_xyz123", always repeat that EXACT name in your answer, never remove underscores or change the format.
4. If the documents list components, features, or steps, reproduce them EXACTLY as they appear, with the SAME numbering, wording, and order.
5. If the DOCUMENTS section does NOT contain relevant information, say "I don't have specific information about this topic in my knowledge base."
6. Avoid making up information not explicitly stated in the documents.
7. If the documents contain contradictions or inaccuracies, report those exactly as given without correction.
8. For follow-up questions, link to information from previous exchanges if relevant.

ANSWER:"""
            else:
                # When no relevant documents are found, use a general LLM prompt
                # that doesn't restrict the model to document information
                system_instruction = "You are a helpful AI assistant that provides accurate and relevant information."
                
                rag_prompt = f"""{system_instruction}

I don't have specific documents in my knowledge base about this topic, but I'll try to help based on my general knowledge.

PREVIOUS CONVERSATION:
{history_context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Provide a helpful and accurate response based on your general knowledge.
2. If the question requires very specific or technical information that you're uncertain about, acknowledge the limitations.
3. For follow-up questions, consider the context from previous exchanges if relevant.

ANSWER:"""

            # Set generation parameters - use very low temperature for tests
            gen_params = {
                "temperature": generation_params.get("temperature", DEFAULT_TEMPERATURE),
                "max_tokens": generation_params.get("max_tokens", DEFAULT_MAX_TOKENS)
            }
            
            # For tests, override with very low temperature
            if 'temperature' in generation_params and generation_params['temperature'] < 0.2:
                gen_params["temperature"] = 0.01  # Force very low temperature for tests
                
            # Add other generation parameters if provided
            for param in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop', 'stream']:
                if param in generation_params:
                    gen_params[param] = generation_params[param]
            
            # Generate the response
            response_text = model_manager.generate(rag_prompt, **gen_params)
            
            # Create response
            response = {
                "answer": response_text,
                "model": model_name,
                "retrieved_documents": results,
                "history_items": history_items if use_history else [],
                "metadata": {
                    "query": query,
                    "document_count": len(results),
                    "history_count": len(history_items),
                    "response_time": time.time() - start_time,
                    "rag_prompt": rag_prompt  # Include the prompt for debugging
                }
            }
            
            # Save to history if enabled and history manager exists
            if use_history and app_history_manager is not None:
                try:
                    app_history_manager.save_response(
                        query=query,
                        response=response_text,
                        metadata={
                            "timestamp": time.time(),
                            "model": model_name,
                            "document_count": len(results),
                            "user_id": search_params.get('user_id', '')  # Include user ID if available
                        }
                    )
                except Exception as history_save_error:
                    logger.warning(f"Failed to save to history: {history_save_error}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG response generation: {str(e)}", exc_info=True)
            raise
