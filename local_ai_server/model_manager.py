import logging
import torch
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class ModelStatus:
    def __init__(self, loaded: bool, model_type: Optional[str] = None, context_window: Optional[int] = None, description: Optional[str] = None):
        self.loaded = loaded
        self.model_type = model_type
        self.context_window = context_window
        self.description = description

class ModelManager:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.model_type = None
        self.context_window = 512  # default context window
        logger.debug(f"Models directory: {self.models_dir}")

    def load_model(self, model_name: str):
        if self.current_model_name == model_name:
            return
        
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            print(f"Loading model from {model_path}")
            if str(model_path).endswith('.gguf'):
                self.model = Llama(
                    model_path=str(model_path),
                    n_ctx=2048,  # Increase context window
                    verbose=False
                )
                self.model_type = 'gguf'
                self.tokenizer = None
                self.context_window = self.model.n_ctx()
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                self.model = AutoModelForCausalLM.from_pretrained(str(model_path))
                self.model_type = 'hf'
        else:
            # Only download for HF models
            print(f"Downloading model {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model_type = 'hf'
            
            print(f"Saving model to {model_path}")
            self.tokenizer.save_pretrained(str(model_path))
            self.model.save_pretrained(str(model_path))
        
        self.current_model_name = model_name

    def list_models(self) -> List[Dict[str, str]]:
        models = []
        logger.debug(f"Scanning models directory: {self.models_dir}")
        
        try:
            # Print absolute path for debugging
            abs_path = self.models_dir.absolute()
            logger.debug(f"Absolute path: {abs_path}")
            logger.debug(f"Directory exists: {abs_path.exists()}")
            logger.debug(f"Is directory: {abs_path.is_dir()}")
            
            # List all files and their details
            if abs_path.exists() and abs_path.is_dir():
                for item in abs_path.iterdir():
                    logger.debug(f"Found item: {item}, is_file: {item.is_file()}")
                    if item.is_file():
                        model_type = 'gguf' if item.name.endswith('.gguf') else 'hf'
                        models.append({
                            "id": item.name,
                            "object": "model",
                            "owned_by": "local",
                            "type": model_type
                        })
            
        except Exception as e:
            logger.error(f"Error scanning models directory: {e}", exc_info=True)
        
        logger.debug(f"Found models: {models}")
        return models

    def get_status(self) -> Dict[str, ModelStatus]:
        models = {}
        
        # First check the models directory for all files
        for model_file in self.models_dir.glob('*'):
            if model_file.is_file():
                model_type = 'gguf' if model_file.name.endswith('.gguf') else 'hf'
                is_loaded = self.current_model_name == model_file.name
                
                # Get description from AVAILABLE_MODELS if it exists
                description = None
                from .models_config import AVAILABLE_MODELS
                if model_file.name in AVAILABLE_MODELS:
                    description = AVAILABLE_MODELS[model_file.name].get('description')
                
                models[model_file.name] = ModelStatus(
                    loaded=is_loaded,
                    model_type=model_type,
                    context_window=self.context_window if is_loaded else None,
                    description=description
                )
        
        return models

    def update_model_info(self, model_name: str, model_type: Optional[str] = None, context_window: Optional[int] = None):
        """Update internal model info cache for dynamic model additions"""
        # This is a helper method to update status for newly uploaded models
        # without requiring a full reload of the model
        model_file = self.models_dir / model_name
        if not model_file.exists():
            logger.warning(f"Cannot update info for non-existent model: {model_name}")
            return
        
        # Force model info to be refreshed on next status check
        logger.info(f"Updated model info for {model_name}: type={model_type}, context={context_window}")

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response to the given prompt using the loaded model."""
        try:
            # Set default parameters if not provided
            max_tokens = kwargs.get('max_tokens', 100)
            temperature = kwargs.get('temperature', 0.7)
            
            if self.model_type == 'gguf':
                # Calculate available tokens
                estimated_prompt_tokens = len(prompt.split())
                if estimated_prompt_tokens + max_tokens > self.context_window:
                    max_tokens = max(0, self.context_window - estimated_prompt_tokens)
                    logger.warning(f"Adjusted max_tokens to {max_tokens} due to context window limits")
                
                # Only pass parameters that the model supports
                model_kwargs = {
                    'prompt': prompt,
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'echo': False
                }
                
                # Add other parameters if provided
                for param in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop']:
                    if param in kwargs and kwargs[param] is not None:
                        model_kwargs[param] = kwargs[param]
                
                response = self.model(**model_kwargs)
                return response['choices'][0]['text']
            else:
                # Handle Hugging Face models
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                # Prepare generate parameters
                generate_kwargs = {
                    'max_length': inputs["input_ids"].shape[1] + max_tokens,
                    'temperature': temperature,
                }
                
                # Add other parameters if provided and supported by HF
                if 'top_p' in kwargs and kwargs['top_p'] is not None:
                    generate_kwargs['top_p'] = kwargs['top_p']
                    
                outputs = self.model.generate(
                    inputs["input_ids"],
                    **generate_kwargs
                )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        except ValueError as e:
            if "context window" in str(e):
                raise RuntimeError(f"Input too long. Maximum context window is {self.context_window} tokens.")
            raise e
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            raise RuntimeError(str(e))

    def generate(self, prompt, **kwargs):
        """Generate a response from the current model"""
        if not self.model:
            raise RuntimeError("No model loaded")
        
        if isinstance(self.model, Llama):
            # For Llama models
            # Only include parameters that were explicitly provided
            completion_params = {
                'prompt': prompt,
                'echo': False
            }
            if 'max_tokens' in kwargs:
                completion_params['max_tokens'] = kwargs['max_tokens']
            if 'temperature' in kwargs:
                completion_params['temperature'] = kwargs['temperature']
            if 'stop' in kwargs:
                completion_params['stop'] = kwargs['stop']
            if 'top_p' in kwargs:
                completion_params['top_p'] = kwargs['top_p']
            if 'frequency_penalty' in kwargs:
                completion_params['frequency_penalty'] = kwargs['frequency_penalty']
            if 'presence_penalty' in kwargs:
                completion_params['presence_penalty'] = kwargs['presence_penalty']

            response = self.model.create_completion(**completion_params)
            return response['choices'][0]['text'].strip()
        else:
            # For Transformers models
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            generate_params = {}
            
            if 'max_tokens' in kwargs:
                generate_params['max_length'] = inputs["input_ids"].shape[1] + kwargs['max_tokens']
            if 'temperature' in kwargs:
                generate_params['temperature'] = kwargs['temperature']
                generate_params['do_sample'] = True
            if 'top_p' in kwargs:
                generate_params['top_p'] = kwargs['top_p']
                generate_params['do_sample'] = True

            # Always include pad_token_id
            generate_params['pad_token_id'] = self.tokenizer.eos_token_id

            outputs = self.model.generate(inputs["input_ids"], **generate_params)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def create_chat_completion(self, messages, **kwargs):
        """Create a chat completion response"""
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        response_text = self.generate(prompt, **kwargs)
        
        return {
            "role": "assistant",
            "content": response_text
        }

# Create global instance
model_manager = ModelManager(Path(__file__).parent / 'models')
