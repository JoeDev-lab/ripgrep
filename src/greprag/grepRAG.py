"""
GrepRAG: Index-Free Lexical Retrieval methodology for codebase completion.
This drop-in module generates codebase grep queries via a local transformers model, 
retrieves structurally merged contexts using ripgrep natively, ranks blocks via 
Tree-Sitter AST (or Regex fallback), and coordinates external inference.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import urllib.request
from typing import Any, Literal
import importlib

logger = logging.getLogger(__name__)

# External dependencies (required)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tree_sitter import Language, Parser, Query


class GrepRAG:
    """
    A drop-in module implementing the GrepRAG methodology for index-free, 
    intent-driven lexical retrieval and generation.
    """

    def __init__(
        self,
        main_inference_url: str,
        main_model_name: str,
        main_model_params: dict[str, Any],
        grep_model_path_or_name: str,
        external_api_type: Literal["openai", "anthropic"],
        external_api_key: str,
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
        context_padding: int = 2
    ) -> None:
        """
        Initializes the GrepRAG pipeline, models, and AST routers.

        Args:
            main_inference_url (str): URL of the external inference server.
            main_model_name (str): Model name for the main external generation.
            main_model_params (dict[str, Any]): Generation parameters (e.g., max_tokens).
            grep_model_path_or_name (str): Path/identifier for the local HuggingFace grep model.
            external_api_type (Literal["openai", "anthropic"]): Schema to use for the external API.
            external_api_key (str): Authentication key for the external main model.
            whitelist (Optional[list[str]]): File/dir wildcard patterns to strictly include.
            blacklist (Optional[list[str]]): File/dir wildcard patterns to exclude.
            context_padding (int): Number of adjacent lines to grab for context block merging.
        """
        self.main_inference_url: str = main_inference_url
        self.main_model_name: str = main_model_name
        self.main_model_params: dict[str, Any] = main_model_params
        self.grep_model_path_or_name: str = grep_model_path_or_name
        
        self.external_api_type: Literal["openai", "anthropic"] = external_api_type
        self.external_api_key: str = external_api_key
        
        self.whitelist: list[str] = whitelist if whitelist is not None else []
        self.blacklist: list[str] = blacklist if blacklist is not None else []
        self._context_padding: int = context_padding
        
        # Internal states for ML and AST dependencies
        self._device: str = "cpu"
        self._tokenizer: Any = None
        self._local_model: Any = None
        
        # Router mappings for tree-sitter AST scoring
        self._language_configs: dict[str, dict[str, str]] = {}
        self._ts_parsers: dict[str, Parser] = {}
        self._ts_queries: dict[str, Query] = {}
        self._missing_lang_warned: set[str] = set()
        
        self._check_dependencies()
        self._init_local_grep_model()
        self._init_tree_sitter()

    def _check_dependencies(self) -> None:
        """
        Verifies that required external system binaries (ripgrep) and parameters are available.
        
        Raises:
            RuntimeError: If 'rg' (ripgrep) is not found in the system PATH.
            ValueError: If an API key is missing.
        """
        try:
            _ = subprocess.run(["rg", "--version"], capture_output=True, check=True)
        except RuntimeError:
            raise RuntimeError("ripgrep ('rg') binary is missing from PATH.")
            
        if not self.external_api_key:
            raise ValueError("An external API key must be provided.")

    def _init_local_grep_model(self) -> None:
        """
        Initializes the local transformers model and tokenizer for grep-query generation.
        Optimized for CUDA if available, falling back to CPU safely.
        """
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.grep_model_path_or_name)
        self._local_model = AutoModelForCausalLM.from_pretrained(
            self.grep_model_path_or_name, 
            torch_dtype=dtype
        ).to(self._device)
        self._local_model.eval()

    def _init_tree_sitter(self) -> None:
        """
        Initializes the dynamic Tree-sitter file extension router with comprehensive language coverage.
        Attempts to load language modules dynamically and pre-compiles their AST queries.
        """
        self._language_configs = {
            ".py": {"module": "tree_sitter_python", "query": "(function_definition name: (identifier) @func_name) (class_definition name: (identifier) @class_name) (identifier) @ident"},
            ".js": {"module": "tree_sitter_javascript", "query": "(function_declaration name: (identifier) @func_name) (class_declaration name: (identifier) @class_name) (identifier) @ident"},
            ".ts": {"module": "tree_sitter_typescript", "query": "(function_declaration name: (identifier) @func_name) (class_declaration name: (type_identifier) @class_name) (identifier) @ident"},
            ".go": {"module": "tree_sitter_go", "query": "(function_declaration name: (identifier) @func_name) (type_spec name: (type_identifier) @class_name) (identifier) @ident"},
            ".rs": {"module": "tree_sitter_rust", "query": "(function_item name: (identifier) @func_name) (struct_item name: (type_identifier) @class_name) (identifier) @ident"},
            ".java": {"module": "tree_sitter_java", "query": "(method_declaration name: (identifier) @func_name) (class_declaration name: (identifier) @class_name) (identifier) @ident"},
            ".cs": {"module": "tree_sitter_c_sharp", "query": "(method_declaration name: (identifier) @func_name) (class_declaration name: (identifier) @class_name) (identifier) @ident"},
            ".cpp": {"module": "tree_sitter_cpp", "query": "(function_definition declarator: (function_declarator declarator: (identifier) @func_name)) (class_specifier name: (type_identifier) @class_name) (identifier) @ident"},
            ".c": {"module": "tree_sitter_c", "query": "(function_definition declarator: (function_declarator declarator: (identifier) @func_name)) (struct_specifier name: (type_identifier) @class_name) (identifier) @ident"},
            ".rb": {"module": "tree_sitter_ruby", "query": "(method name: (identifier) @func_name) (class name: (constant) @class_name) (identifier) @ident"},
            ".php": {"module": "tree_sitter_php", "query": "(function_definition name: (name) @func_name) (class_declaration name: (name) @class_name) (name) @ident"}
        }

        for ext, config in self._language_configs.items():
            try:
                lang_module = importlib.import_module(config["module"])
                # Note: Handles `language()` export pattern standard in tree-sitter v0.21+
                ts_language = Language(lang_module.language())
                parser = Parser()
                parser.language = ts_language
                
                self._ts_queries[ext] = ts_language.query(config["query"])
                self._ts_parsers[ext] = parser
            except ImportError:
                # Silently fail on init; we will log a warning during inference if the file is encountered
                pass
            except Exception:
                logger.error("Failed to load configured tree-sitter for %s: %s", ext, sys.exc_info()[1])

    def _generate_grep_queries(self, prompt: str) -> list[str]:
        """
        Prompts the local specialized model to output a JSON list of regex queries.
        
        Args:
            prompt (str): The user's codebase query.
            
        Returns:
            list[str]: A list of regex strings.
        """
        system_prompt: str = "Generate a JSON list of short, specific regex strings to search the codebase for the user's intent. Output ONLY the JSON array of strings."
        full_prompt: str = f"{system_prompt}\n\nUser: {prompt}\nRegex Queries:"
        
        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._device)
        
        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs, 
                max_new_tokens=128, 
                temperature=0.1, 
                pad_token_id=self._tokenizer.eos_token_id
            )
            
        response: str = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
        
        try:
            clean_json: str = response.strip().strip("`").removeprefix("json").strip()
            queries = json.loads(clean_json)
            if isinstance(queries, list):
                return [str(q) for q in queries]
        except json.JSONDecodeError:
            return [line.strip() for line in response.splitlines() if line.strip()]
            
        return []

    def _run_ripgrep(self, queries: list[str], repo_path: str) -> list[dict[str, Any]]:
        """
        Executes a single ripgrep pass, letting the Rust binary natively deduplicate overlapping windows.
        
        Args:
            queries (list[str]): Regex queries to evaluate simultaneously.
            repo_path (str): Codebase root directory.
            
        Returns:
            list[dict[str, Any]]: Structural blocks with file metadata and lines.
        """
        if not queries:
            return []
            
        cmd: list[str] = ["rg", "--json", "-C", str(self._context_padding)]
        
        for q in queries:
            cmd.extend(["-e", q])
            
        for wl in self.whitelist:
            cmd.extend(["-g", wl])
        for bl in self.blacklist:
            cmd.extend(["-g", f"!{bl}"])
            
        cmd.append(repo_path)
        
        blocks: list[dict[str, Any]] = []
        current_block: dict[str, Any] | None = None
        last_line_num: int = -1
        
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)
            for line in process.stdout.splitlines():
                if not line.strip():
                    continue
                try:
                    parsed: dict[str, Any] = json.loads(line)
                    msg_type: str = parsed.get("type", "")
                    
                    if msg_type in ("match", "context"):
                        data: dict[str, Any] = parsed.get("data", {})
                        file_path: str = data.get("path", {}).get("text", "")
                        line_num: int = data.get("line_number", 0)
                        content: str = data.get("lines", {}).get("text", "").rstrip("\n")
                        
                        if current_block is None or current_block["file"] != file_path or line_num > last_line_num + 1:
                            if current_block is not None:
                                blocks.append(current_block)
                            current_block = {
                                "file": file_path,
                                "start_line": line_num,
                                "end_line": line_num,
                                "lines": [content]
                            }
                        else:
                            current_block["lines"].append(content)
                            current_block["end_line"] = line_num
                            
                        last_line_num = line_num
                except json.JSONDecodeError:
                    continue
                    
            if current_block is not None:
                blocks.append(current_block)
                
        except Exception:
            logger.error("Ripgrep execution failed: %s", sys.exc_info()[1])
            
        return blocks

    async def _run_ripgrep_async(self, queries: list[str], repo_path: str) -> list[dict[str, Any]]:
        """
        Executes a single ripgrep pass asynchronously, letting the Rust binary natively deduplicate overlapping windows.
        
        Args:
            queries (list[str]): Regex queries to evaluate simultaneously.
            repo_path (str): Codebase root directory.
            
        Returns:
            list[dict[str, Any]]: Structural blocks with file metadata and lines.
        """
        return await asyncio.to_thread(self._run_ripgrep, queries, repo_path)

    def _regex_fallback_score(self, content: str) -> float:
        """
        A heuristic scorer for unknown file extensions or uninstalled AST libraries.
        
        Args:
            content (str): Raw string code block.
            
        Returns:
            float: Identifier significance score.
        """
        score: float = 0.0
        if re.search(r'\b(class|def|function|struct|interface|type)\b', content):
            score += 5.0
            
        identifiers: list[str] = re.findall(r'\b[a-zA-Z_]\w*\b', content)
        for ident in identifiers:
            if len(ident) > 3 and ident not in {"self", "this", "True", "False", "None"}:
                score += 0.5
        return score

    def _ast_weighted_rerank(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Uses Tree-sitter parsing (or regex fallback) to score block significance.
        Logs a single warning per file-type if a supported language module is missing.
        
        Args:
            blocks (list[dict[str, Any]]): Structural code blocks.
            
        Returns:
            list[dict[str, Any]]: Blocks sorted by significance (descending).
        """
        for block in blocks:
            file_name: str = block.get("file", "")
            _, ext = os.path.splitext(file_name)
            ext = ext.lower()
            code_text: bytes = bytes("\n".join(block["lines"]), "utf8")
            
            score: float = 0.0
            
            if ext in self._ts_parsers and ext in self._ts_queries:
                parser: Parser = self._ts_parsers[ext]
                query: Query = self._ts_queries[ext]
                
                tree = parser.parse(code_text)
                captures = query.captures(tree.root_node)
                
                for node, capture_name in captures:
                    if capture_name in ("func_name", "class_name"):
                        score += 5.0
                    elif capture_name == "ident":
                        ident_text: str = node.text.decode("utf8")
                        if len(ident_text) > 3 and ident_text not in {"self", "this", "True", "False", "None"}:
                            score += 0.5
            else:
                # Log warning if we mapped this extension but the user didn't pip install the module
                if ext in self._language_configs and ext not in self._missing_lang_warned:
                    module_name = self._language_configs[ext]["module"]
                    logger.warning("GrepRAG: '%s' is not installed. Falling back to Regex scoring for '%s' files.", module_name, ext)
                    self._missing_lang_warned.add(ext)
                    
                score = self._regex_fallback_score("\n".join(block["lines"]))
                
            block["score"] = score
            
        return sorted(blocks, key=lambda x: x.get("score", 0.0), reverse=True)

    def _format_context(self, blocks: list[dict[str, Any]]) -> str:
        """
        Formats codebase blocks into a prompt-friendly string structure.
        """
        context_str: str = ""
        for block in blocks:
            context_str += f"\n--- File: {block['file']} (Lines {block['start_line']}-{block['end_line']}) ---\n"
            context_str += "\n".join(block["lines"])
            context_str += "\n...\n"
        return context_str

    def _call_external_inference(self, prompt: str, context: str) -> str:
        """Routes prompt and context to the configured LLM API."""
        if self.external_api_type == "openai":
            return self._call_openai(prompt, context)
        elif self.external_api_type == "anthropic":
            return self._call_anthropic(prompt, context)
        else:
            raise ValueError("Unsupported external API type.")

    def _call_openai(self, prompt: str, context: str) -> str:
        """OpenAI-compatible REST execution via pure stdlib urllib."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.external_api_key}"
        }
        
        payload: dict[str, Any] = {
            "model": self.main_model_name,
            "messages": [
                {"role": "system", "content": f"Use the following codebase context to assist the user:\n{context}"},
                {"role": "user", "content": prompt}
            ],
            **self.main_model_params
        }
        
        req = urllib.request.Request(self.main_inference_url, headers=headers, data=json.dumps(payload).encode("utf-8"))
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except Exception as exc:
            return f"Error calling OpenAI API: {exc!r}"

    def _call_anthropic(self, prompt: str, context: str) -> str:
        """Anthropic-compatible REST execution via pure stdlib urllib."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-api-key": self.external_api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload: dict[str, Any] = {
            "model": self.main_model_name,
            "system": f"Use the following codebase context to assist the user:\n{context}",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            **self.main_model_params
        }
        
        req = urllib.request.Request(self.main_inference_url, headers=headers, data=json.dumps(payload).encode("utf-8"))
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["content"][0]["text"]
        except Exception as exc:
            return f"Error calling Anthropic API: {exc!r}"

    def process(self, prompt: str, repo_path: str, top_k: int = 20) -> str:
        """
        The main pipeline execution: Generates intents, executes grep, scores ASTs, and yields inference.
        
        Args:
            prompt (str): The user's codebase intent.
            repo_path (str): Local codebase root path.
            top_k (int): Number of top-ranked context blocks to relay to the LLM.
            
        Returns:
            str: External LLM response.
        """
        queries: list[str] = self._generate_grep_queries(prompt)
        blocks: list[dict[str, Any]] = self._run_ripgrep(queries, repo_path)
        ranked_blocks: list[dict[str, Any]] = self._ast_weighted_rerank(blocks)
        
        formatted_context: str = self._format_context(ranked_blocks[:top_k])
        return self._call_external_inference(prompt, formatted_context)

    async def process_async(self, prompt: str, repo_path: str, top_k: int = 20) -> str:
        """
        The main pipeline execution asynchronously: Generates intents, executes grep, scores ASTs, and yields inference.
        
        Args:
            prompt (str): The user's codebase intent.
            repo_path (str): Local codebase root path.
            top_k (int): Number of top-ranked context blocks to relay to the LLM.
            
        Returns:
            str: External LLM response.
        """
        queries: list[str] = self._generate_grep_queries(prompt)
        blocks: list[dict[str, Any]] = await self._run_ripgrep_async(queries, repo_path)
        ranked_blocks: list[dict[str, Any]] = self._ast_weighted_rerank(blocks)
        
        formatted_context: str = self._format_context(ranked_blocks[:top_k])
        return self._call_external_inference(prompt, formatted_context)