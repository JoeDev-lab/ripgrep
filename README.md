# GrepRAG: Index-Free Lexical Retrieval for Coding Agents

A lightweight, drop-in Python module and FastMCP server implementing the **GrepRAG** methodology. It provides index-free, intent-driven codebase retrieval for local AI coding agents without the heavy overhead, cold-start delays, or context fragmentation of traditional Vector RAG systems.

This project is a standalone, production-ready implementation inspired by the academic research behind GrepRAG, optimized for local agents and Model Context Protocol (MCP) clients.

## Acknowledgements & Resources
* **Original Paper:** [GrepRAG: Injecting Exact Lexical Context for Code Generation](<https://arxiv.org/abs/2406.14497>)
* **Official Academic Repo:** [ZJU-ACES-ISE/greprag](<https://github.com/ZJU-ACES-ISE/greprag>)
* **Specialized Grep Model:** [`greprag0/greprag-0.6b`](<https://huggingface.co/greprag0/greprag-0.6b>) (Distilled 0.6B Qwen model fine-tuned to output targeted regex queries).

---

## How GrepRAG Works (Step-by-Step)

Traditional Vector RAG struggles with code because it relies on fuzzy semantic proximity, often slicing syntax trees into arbitrary chunks. GrepRAG behaves like an experienced developer using exact lexical and structural pattern matching across files:

1. **Intent-Driven Query Generation:** 
   Instead of embedding the prompt, GrepRAG sends the request to a small, specialized local model (e.g., `greprag-0.6b`). The model outputs a targeted JSON array of regex search strings (e.g., `["def validate_payment", "class PaymentHandler"]`).
2. **Multi-Query Lexical Retrieval (`ripgrep`):**
   The queries are evaluated simultaneously in a single `ripgrep` subprocess. Ripgrep’s Rust engine scans the repository natively, enforces include/exclude globs at the C/Rust level, captures surrounding line context, and deduplicates overlapping match ranges.
3. **AST-Aware Re-ranking (Tree-sitter):**
   Grep results can include noisy matches (e.g., common identifiers like `id` or `config`). The pipeline parses matched blocks using language-specific Tree-sitter parsers, scoring them by syntactic importance (prioritizing class definitions, function signatures, and rare identifiers). If a specific language parser is not installed, it falls back to a regex heuristic scorer with a single warning.
4. **Context Formatting & Inference:**
   The top-scoring structural code blocks are merged into formatted snippets and sent alongside the user prompt to an external LLM (OpenAI or Anthropic compatible) to produce the final completion.

---

## Installation

### 1. System Dependency
The `ripgrep` executable must be available on your system `PATH`.
* **Ubuntu/Debian:** `sudo apt-get install ripgrep`
* **macOS:** `brew install ripgrep`
* **Windows:** `choco install ripgrep`

Verify installation:
```bash
rg --version
```

### 2. Python Package Installation
Install directly from the source repository:
```bash
# Minimal install (core engine + regex fallback scoring)
pip install .

# Recommended: Install with all supported Tree-sitter language parsers
pip install ".[all-languages]"
```

# Granular Language Installs
If you want to keep dependencies small, you can install only the languages your project needs:
```bash
# Install specific language parsers
pip install ".[python,typescript,go]"
```
Available extras: python, javascript, typescript, go, rust, java, c-sharp, cpp, c, ruby, php, and all-languages.

# Development Install
```bash
pip install -e ".[all-languages,dev]"
```

### Usage

#### As a Python Library

```python
from greprag import GrepRAG

# Initialize pipeline
rag = GrepRAG(
    main_inference_url="https://api.openai.com/v1/chat/completions",
    main_model_name="gpt-4o",
    main_model_params={"temperature": 0.2, "max_tokens": 1024},
    grep_model_path_or_name="greprag0/greprag-0.6b",
    external_api_type="openai",
    external_api_key="sk-your-api-key",
    whitelist=["*.py", "*.ts"],           # Optional: include globs
    blacklist=["node_modules", ".git"],     # Optional: exclude globs
    context_padding=3                     # Adjacent lines to merge
)

# Run retrieval + generation on a local repository
prompt = "Where is webhook signature verification implemented, and how are expired timestamps handled?"
repo_directory = "/path/to/your/codebase"

response = rag.process(
    prompt=prompt,
    repo_path=repo_directory,
    top_k=15
)

print(response)
```

**Async usage:** `rag.process_async()` is an async wrapper supporting the same parameters; use it in async contexts with `await` or via `asyncio.run()`.

**CUDA support:** The local grep model automatically uses GPU acceleration if available — no extra configuration needed. Performance on large repos may be noticeably better on CUDA devices.

---

#### Running as an MCP Server

Once installed, the package provides a console script:

```bash
# Set required credentials
export GREPRAG_EXTERNAL_API_KEY="sk-your-api-key"
export GREPRAG_EXTERNAL_API_TYPE="openai"
export GREPRAG_MAIN_MODEL_NAME="gpt-4o"

# Run the FastMCP server
greprag-server
```

**Default blacklist patterns:** `.git`, `node_modules`, `venv`, `__pycache__`, and `.pytest_cache` are excluded by default — you can customize them via `GREPRAG_BLACKLIST` or set `blacklist=[]` during initialization.

---

#### Claude Desktop Configuration (claude_desktop_config.json)

Add the server to your configuration:
```json
{
  "mcpServers": {
    "greprag": {
      "command": "greprag-server",
      "env": {
        "GREPRAG_EXTERNAL_API_KEY": "sk-your-api-key",
        "GREPRAG_EXTERNAL_API_TYPE": "openai",
        "GREPRAG_MAIN_MODEL_NAME": "gpt-4o",
        "GREPRAG_GREP_MODEL_PATH": "greprag0/greprag-0.6b"
      }
    }
  }
}
```

## Running as an MCP Server

Once installed, the package provides a console script:

```bash
# Set required credentials
export GREPRAG_EXTERNAL_API_KEY="sk-your-api-key"
export GREPRAG_EXTERNAL_API_TYPE="openai"
export GREPRAG_MAIN_MODEL_NAME="gpt-4o"

# Run the FastMCP server
greprag-server
```

### Claude Desktop Configuration (claude_desktop_config.json)

Add the server to your configuration:
```json
{
  "mcpServers": {
    "greprag": {
      "command": "greprag-server",
      "env": {
        "GREPRAG_EXTERNAL_API_KEY": "sk-your-api-key",
        "GREPRAG_EXTERNAL_API_TYPE": "openai",
        "GREPRAG_MAIN_MODEL_NAME": "gpt-4o",
        "GREPRAG_GREP_MODEL_PATH": "greprag0/greprag-0.6b"
      }
    }
  }
}
```

# Exposed MCP Tools

- **`greprag_query_and_answer(prompt, repo_path, top_k)`**: Full retrieval pipeline (regex query generation → ripgrep → AST-ranked results) followed by external LLM inference. The returned context is sent to the model together with your prompt — expect token costs proportional to `top_k` blocks (each block includes file path, line numbers, and code content). Use lower `top_k` values for tighter cost control.

- **`greprag_retrieve_context(prompt, repo_path, top_k)`**: Pure retrieval step only. Runs the same local pipeline (regex → ripgrep → Tree-sitter ranking) but returns formatted code blocks without calling the external model. Ideal when you want to inspect or further process results yourself; no LLM token consumption from GrepRAG side.

---

### Environment Variables Reference

| `GREPRAG_MAIN_INFERENCE_URL` | https://api.openai.com/v1/chat/completions | Inference endpoint URL |
| `GREPRAG_EXTERNAL_API_KEY` | *(required)* | API bearer token (OpenAI) or x-api-key (Anthropic) |
| `GREPRAG_GREP_MODEL_PATH` | greprag0/greprag-0.6b | Local path or Hugging Face model identifier for the regex-generation model |
| `GREPRAG_WHITELIST` | *(empty)* | Comma-separated include globs (e.g., `*.py,*.ts`) — overrides instance settings in server mode |
| `GREPRAG_BLACKLIST` | `.git,node_modules,venv,__pycache__,.pytest_cache` | Comma-separated exclude globs — overrides instance settings in server mode |
| `GREPRAG_CONTEXT_PADDING` | 2 | Lines of adjacent context to include per match (passed as `-C` to ripgrep) |
| `GREPRAG_EXTERNAL_API_KEY` | *(required)* | API bearer token (OpenAI) or x-api-key (Anthropic) |
| `GREPRAG_WHITELIST` | *(empty)* | Comma-separated include globs (e.g., `*.py,*.ts`) — overrides instance settings in server mode |
| `GREPRAG_BLACKLIST` | `.git,node_modules,venv,__pycache__,.pytest_cache` | Comma-separated exclude globs — overrides instance settings in server mode |
| `GREPRAG_CONTEXT_PADDING` | 2 | Lines of adjacent context to include per match (passed as `-C` to ripgrep) |

*Note: When using the Python API, you can customize these via constructor arguments; the environment variables only affect the MCP server.*