"""
FastMCP Server for GrepRAG.
Exposes index-free lexical retrieval and code generation as Model Context Protocol tools.
Configuration is loaded entirely from environment variables.
"""

import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional

from mcp.server.fastmcp import FastMCP

from greprag import GrepRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Environment Variable Configuration ---
MAIN_INFERENCE_URL: str = os.getenv("GREPRAG_MAIN_INFERENCE_URL", "https://api.openai.com/v1/chat/completions")
MAIN_MODEL_NAME: str = os.getenv("GREPRAG_MAIN_MODEL_NAME", "gpt-4o")
MAIN_MODEL_PARAMS_RAW: str = os.getenv("GREPRAG_MAIN_MODEL_PARAMS", '{"temperature": 0.2, "max_tokens": 2048}')
GREP_MODEL_PATH: str = os.getenv("GREPRAG_GREP_MODEL_PATH", "greprag0/greprag-0.6b")

EXTERNAL_API_TYPE_RAW: str = os.getenv("GREPRAG_EXTERNAL_API_TYPE", "openai").lower()
EXTERNAL_API_TYPE: Literal["openai", "anthropic"] = "anthropic" if EXTERNAL_API_TYPE_RAW == "anthropic" else "openai"
EXTERNAL_API_KEY: str = os.getenv("GREPRAG_EXTERNAL_API_KEY", "")

# Lists parsed from comma-separated strings
WHITELIST_RAW: str = os.getenv("GREPRAG_WHITELIST", "")
BLACKLIST_RAW: str = os.getenv("GREPRAG_BLACKLIST", ".git,node_modules,venv,__pycache__,.pytest_cache")
CONTEXT_PADDING: int = int(os.getenv("GREPRAG_CONTEXT_PADDING", "2"))

WHITELIST: Optional[List[str]] = [w.strip() for w in WHITELIST_RAW.split(",") if w.strip()] or []
BLACKLIST: Optional[List[str]] = [b.strip() for b in BLACKLIST_RAW.split(",") if b.strip()] or []

try:
    MAIN_MODEL_PARAMS: Dict[str, Any] = json.loads(MAIN_MODEL_PARAMS_RAW)
except json.JSONDecodeError:
    logging.warning("Failed to parse GREPRAG_MAIN_MODEL_PARAMS as JSON. Defaulting to empty dict.")
    MAIN_MODEL_PARAMS: Dict[str, Any] = {}

# --- Initialize GrepRAG Pipeline ---
logging.info("Initializing GrepRAG core engine...")
rag_instance = GrepRAG(
    main_inference_url=MAIN_INFERENCE_URL,
    main_model_name=MAIN_MODEL_NAME,
    main_model_params=MAIN_MODEL_PARAMS,
    grep_model_path_or_name=GREP_MODEL_PATH,
    external_api_type=EXTERNAL_API_TYPE,
    external_api_key=EXTERNAL_API_KEY,
    whitelist=WHITELIST,
    blacklist=BLACKLIST,
    context_padding=CONTEXT_PADDING
)

# --- MCP Server Definition ---
mcp = FastMCP("GrepRAG-Server")


@mcp.tool()
def greprag_query_and_answer(prompt: str, repo_path: str, top_k: int = 15) -> str:
    """
    Searches a local repository using GrepRAG and passes the context to the main LLM to generate an answer.

    Args:
        prompt: The question or programming task to address.
        repo_path: Absolute path to the codebase on disk.
        top_k: Number of highest-ranked AST code blocks to forward as context.

    Returns:
        The generated response from the external language model.
    """
    if not os.path.isdir(repo_path):
        return f"Error: Repository path '{repo_path}' does not exist or is not a directory."

    try:
        return rag_instance.process(prompt=prompt, repo_path=repo_path, top_k=top_k)
    except Exception as e:
        logging.error(f"Error executing GrepRAG pipeline: {e}")
        return f"Error processing query: {str(e)}"


@mcp.tool()
def greprag_retrieve_context(prompt: str, repo_path: str, top_k: int = 15) -> str:
    """
    Searches a local repository using GrepRAG and returns ONLY the structured code snippets.
    Useful when the client agent prefers to handle the reasoning itself without calling an external model.

    Args:
        prompt: The context or code symbols to look for.
        repo_path: Absolute path to the codebase on disk.
        top_k: Number of highest-ranked AST code blocks to return.

    Returns:
        Formatted code blocks with file names and line numbers.
    """
    if not os.path.isdir(repo_path):
        return f"Error: Repository path '{repo_path}' does not exist or is not a directory."

    try:
        queries = rag_instance._generate_grep_queries(prompt)
        blocks = rag_instance._run_ripgrep(queries, repo_path)
        ranked = rag_instance._ast_weighted_rerank(blocks)
        return rag_instance._format_context(ranked[:top_k])
    except Exception as e:
        logging.error(f"Error retrieving context: {e}")
        return f"Error retrieving context: {str(e)}"


def main() -> None:
    """Entrypoint for the MCP command line runner."""
    mcp.run()


if __name__ == "__main__":
    main()