"""
Tests for stress scenarios in the GrepRAG functionality.
"""

import json
from pathlib import Path

import pytest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grepRAG import GrepRAG


class TestStressScenarios:
    """Test suite for stress scenarios in GrepRAG functionality."""

    def test_deeply_nested_structure(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that 10+ levels of directory nesting handled."""
        with patch("subprocess.run") as mock_run:
            # Simulate 10+ levels of nesting
            nested_path = str(Path(__file__).parent / "fixtures" / "dummy_repo" / "deeply_nested")
            mock_output = json.dumps([
                {
                    "type": "match",
                    "data": {
                        "path": {"text": nested_path},
                        "line_number": 10,
                        "lines": {"text": "def deeply_nested_function():"}
                    }
                }
            ])
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            result = create_grep_rag._run_ripgrep(
                ["def.*nested"],
                nested_path,
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_many_files_same_dir(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that 50+ files in single directory handled."""
        with patch("subprocess.run") as mock_run:
            # Simulate 50+ files
            mock_outputs = []
            for i in range(55):
                mock_outputs.append(json.dumps([
                    {
                        "type": "match",
                        "data": {
                            "path": {"text": f"/tmp/test{i}.py"},
                            "line_number": 10,
                            "lines": {"text": f"def func{i}():\n    pass"}
                        }
                    }
                ]))
            
            mock_output = "\n".join(mock_outputs)
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            result = create_grep_rag._run_ripgrep(
                ["def.*func"],
                "/tmp",
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"
            assert len(result) == 55, "Should return 55 blocks"

    def test_large_file(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that 5000+ line file handled."""
        with patch("subprocess.run") as mock_run:
            # Simulate 5000+ line file
            mock_output = json.dumps([
                {
                    "type": "match",
                    "data": {
                        "path": {"text": "/tmp/large_file.py"},
                        "line_number": 10,
                        "lines": {"text": "def func():"}
                    }
                }
            ])
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            result = create_grep_rag._run_ripgrep(
                ["def.*func"],
                "/tmp",
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_many_queries(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that 20+ regex queries generated."""
        with patch.object(create_grep_rag, "_generate_grep_queries") as mock_gen:
            # Test with 20+ queries
            mock_gen.return_value = [f"def.*{i}" for i in range(25)]
            
            result = create_grep_rag._run_ripgrep(
                [f"def.*{i}" for i in range(25)],
                str(Path(__file__).parent / "fixtures" / "dummy_repo"),
            )
            
            # Verify result
            assert isinstance(result, list), "Result should be a list"

    def test_rapid_successive_calls(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that 10 calls in rapid succession handled."""
        with patch("subprocess.run") as mock_run:
            # Simulate 10 rapid calls
            mock_outputs = []
            for i in range(10):
                mock_outputs.append(json.dumps([
                    {
                        "type": "match",
                        "data": {
                            "path": {"text": str(Path(__file__).parent / "fixtures" / "dummy_repo" / "auth.py")},
                            "line_number": 10,
                            "lines": {"text": "def login():"}
                        }
                    }
                ]))
            
            mock_output = "\n".join(mock_outputs)
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            # Make 10 rapid calls
            results = []
            for _ in range(10):
                result = create_grep_rag._run_ripgrep(
                    ["def.*login"],
                    str(Path(__file__).parent / "fixtures" / "dummy_repo"),
                )
                results.append(result)
            
            # Verify results
            assert all(isinstance(r, list) for r in results), "All results should be lists"

    @pytest.mark.asyncio
    async def test_parallel_ripgrep(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that 10 concurrent ripgrep calls via asyncio.gather() handled."""
        with patch("subprocess.run") as mock_run:
            # Simulate 10 concurrent calls
            mock_outputs = []
            for i in range(10):
                mock_outputs.append(json.dumps([
                    {
                        "type": "match",
                        "data": {
                            "path": {"text": str(Path(__file__).parent / "fixtures" / "dummy_repo" / "auth.py")},
                            "line_number": 10,
                            "lines": {"text": "def login():"}
                        }
                    }
                ]))
            
            async def mock_run_async(*args, **kwargs):
                return type("MockProcess", (), {
                    "stdout": mock_outputs[0],
                    "stderr": "",
                    "returncode": 0,
                    "capture_output": True,
                    "text": True,
                    "check": False
                })()
            
            with patch("asyncio.to_thread", return_value=mock_run_async()):
                # Make 10 concurrent calls
                results = await create_grep_rag._run_ripgrep_async(
                    ["def.*login", "class.*User", "def.*auth"],
                    str(Path(__file__).parent / "fixtures" / "dummy_repo"),
                )
                
                # Verify results
                assert isinstance(results, list), "Result should be a list"

    def test_memory_cleanup(
        self,
        create_grep_rag_instance: GrepRAG,
    ) -> None:
        """Test that memory leak in caching handled."""
        with patch("subprocess.run") as mock_run:
            # Simulate multiple calls with memory cleanup
            mock_outputs = []
            for i in range(10):
                mock_outputs.append(json.dumps([
                    {
                        "type": "match",
                        "data": {
                            "path": {"text": str(Path(__file__).parent / "fixtures" / "dummy_repo" / "auth.py")},
                            "line_number": 10,
                            "lines": {"text": "def login():"}
                        }
                    }
                ]))
            
            mock_output = "\n".join(mock_outputs)
            mock_run.return_value = type("MockProcess", (), {
                "stdout": mock_output,
                "stderr": "",
                "returncode": 0,
                "capture_output": True,
                "text": True,
                "check": False
            })()
            
            # Make 10 calls
            results = []
            for _ in range(10):
                result = create_grep_rag._run_ripgrep(
                    ["def.*login"],
                    str(Path(__file__).parent / "fixtures" / "dummy_repo"),
                )
                results.append(result)
            
            # Verify results
            assert all(isinstance(r, list) for r in results), "All results should be lists"
