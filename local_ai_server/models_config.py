"""Configuration for available models and their metadata."""

# Add the EMBEDDING_MODEL constant - importing from endpoints.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

AVAILABLE_MODELS = {
    "phi-2.Q4_K_M.gguf": {
        "name": "Phi-2 (4-bit Quantized)",
        "description": "Microsoft's Phi-2 model optimized for efficient inference",
        "size": "2.3GB",
        "type": "gguf",
        "context_window": 2048,
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
    },
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf": {
        "name": "TinyLlama Chat",
        "description": "Lightweight chat model optimized for efficiency",
        "size": "1.1GB",
        "type": "gguf",
        "context_window": 2048,
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    },
    "neural-chat-7b-v3-1.Q4_K_M.gguf": {
        "name": "Neural Chat v3.1",
        "description": "Optimized chat model with good performance",
        "size": "4.3GB",
        "type": "gguf",
        "context_window": 4096,
        "url": "https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_K_M.gguf",
    },
    "openhermes-2.5-mistral-7b.Q4_K_M.gguf": {
        "name": "OpenHermes 2.5 Mistral (4-bit)",
        "description": "High quality open source model based on Mistral 7B",
        "size": "4.37GB",
        "type": "gguf",
        "context_window": 4096,
        "url": "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf",
    },
    "openchat-3.5-0106.Q4_K_M.gguf": {
        "name": "OpenChat 3.5",
        "description": "State-of-the-art open source chat model",
        "size": "4.31GB",
        "type": "gguf",
        "context_window": 4096,
        "url": "https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q4_K_M.gguf",
    },
    "stablelm-zephyr-3b.Q4_K_M.gguf": {
        "name": "StableLM Zephyr 3B",
        "description": "Lightweight and efficient chat model",
        "size": "1.93GB",
        "type": "gguf",
        "context_window": 4096,
        "url": "https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF/resolve/main/stablelm-zephyr-3b.Q4_K_M.gguf",
    },
    "gemma-2b.Q4_K_M.gguf": {
        "name": "Gemma 2B",
        "description": "Google's lightweight open model",
        "size": "1.53GB",
        "type": "gguf",
        "context_window": 4096,
        "url": "https://huggingface.co/TheBloke/Gemma-2b-GGUF/resolve/main/gemma-2b.Q4_K_M.gguf",
    }
}

MODEL_DEFAULTS = {
    "gguf": {
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": ["User:", "Assistant:"]
    },
    "hf": {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True
    }
}
