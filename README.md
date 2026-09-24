# GrepRAG

**Index-Free Lexical Retrieval** methodology for codebase completion.

## Overview

GrepRAG is a drop-in Python module that generates codebase grep queries via a local transformers model, retrieves structurally merged contexts using ripgrep natively, ranks blocks via Tree-Sitter AST (or Regex fallback), and coordinates external inference.

#TODO: Make 'paper' and 'repo' hyperlinks for the addresses below.
Implemented from original paper and repo:
  https://arxiv.org/html/2601.23254v2
  https://github.com/ZJU-ACES-ISE/greprag

#TODO: Make 'model' hyperlink to address.
pretrained greprag model:
  https://huggingface.co/greprag0/greprag-0.6b
  

## Key Features

- **Intent-Driven Retrieval**: Uses an LLM to generate regex queries from natural language prompts
- **Native Ripgrep**: Leverages ripgrep's speed and JSON output for efficient codebase scanning
- **AST-Based Ranking**: Uses Tree-Sitter to parse and score code blocks by structural significance
- **Fallback Scoring**: Gracefully falls back to regex-based scoring for unknown file types
- **Multi-Language Support**: Pre-configured for Python, JavaScript/TypeScript, Go, Rust, Java, C#, C++, Ruby, PHP, and more

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure `ripgrep` (rg) is installed and in your PATH:

```bash
# Check if ripgrep is installed
rg --version
```

## Usage

### Basic Example

```python
from grepRAG import GrepRAG

# Initialize the pipeline
rag = GrepRAG(
    main_inference_url="https://your-inference-endpoint.com/v1/chat/completions",
    main_model_name="your-main-model-name",
    main_model_params={"max_tokens": 500, "temperature": 0.7},
    grep_model_path_or_name="your-local-model-or-hf-id",
    external_api_type="openai",
    external_api_key="your-api-key",
    context_padding=2
)

# Process a query
result = rag.process(
    prompt="Find all functions related to user authentication",
    repo_path="/path/to/your/codebase",
    top_k=20
)

print(result)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `main_inference_url` | str | - | URL of the external inference server |
| `main_model_name` | str | - | Model name for the main external generation |
| `main_model_params` | dict | - | Generation parameters (e.g., `max_tokens`, `temperature`) |
| `grep_model_path_or_name` | str | - | Path/identifier for the local HuggingFace grep model |
| `external_api_type` | Literal["openai", "anthropic"] | - | Schema to use for the external API |
| `external_api_key` | str | - | Authentication key for the external main model |
| `whitelist` | Optional[list[str]] | None | File/dir wildcard patterns to strictly include |
| `blacklist` | Optional[list[str]] | None | File/dir wildcard patterns to exclude |
| `context_padding` | int | 2 | Number of adjacent lines to grab for context block merging |

## Supported Languages

The following languages are supported via Tree-Sitter parsers:

- **Python** (`tree-sitter-python`)
- **JavaScript/TypeScript** (`tree-sitter-javascript`, `tree-sitter-typescript`)
- **Go** (`tree-sitter-go`)
- **Rust** (`tree-sitter-rust`)
- **Java** (`tree-sitter-java`)
- **C#** (`tree-sitter-c-sharp`)
- **C++** (`tree-sitter-cpp`)
- **C** (`tree-sitter-c`)
- **Ruby** (`tree-sitter-ruby`)
- **PHP** (`tree-sitter-php`)

**Note**: If a Tree-Sitter parser isn't installed for a specific language, GrepRAG will fall back to regex-based scoring with a warning logged.

## Project Structure

```
grepRAG/
├── __init__.py          # Package initialization
├── grepRAG.py           # Main GrepRAG module
├── requirements.txt     # Python dependencies
├── example_usage.py     # Example usage script
├── README.md            # This file
└── .gitignore           # Git ignore patterns
```

## Dependencies

- **Core**:
  - `torch` (PyTorch)
  - `transformers` (HuggingFace)
  - `tree-sitter`
  
- **Language Parsers** (optional, for AST-based ranking):
  - `tree-sitter-python`, `tree-sitter-javascript`, etc.

- **System**:
  - `ripgrep` (rg) - must be installed separately

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
