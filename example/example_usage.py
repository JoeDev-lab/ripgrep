"""
Example usage of the GrepRAG module.
"""

from grepRAG import GrepRAG

# Initialize the GrepRAG pipeline
# Replace with your local model path or HuggingFace model name
GREP_MODEL = "your-local-model-path-or-hf-id"

# External inference configuration
GREP_RAG = GrepRAG(
    main_inference_url="https://your-inference-endpoint.com/v1/chat/completions",
    main_model_name="your-main-model-name",
    main_model_params={
        "max_tokens": 500,
        "temperature": 0.7,
    },
    grep_model_path_or_name=GREP_MODEL,
    external_api_type="openai",  # or "anthropic"
    external_api_key="your-api-key",
    whitelist=None,  # Optional: ["**/*"],
    blacklist=None,  # Optional: ["**/node_modules/**"],
    context_padding=2
)

# Process a query against your codebase
result = GREP_RAG.process(
    prompt="Find all functions related to user authentication",
    repo_path="/path/to/your/codebase",
    top_k=20
)

print(result)
