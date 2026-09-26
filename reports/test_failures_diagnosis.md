# Test Suite Failure Report

## Summary
Total tests: 71
- **Passed:** 54
- **Failed:** 17  
- **Errors during collection:** 1 (syntax error)

---

## Category 1: Syntax Error (Prevents Test Collection)

### `test_query_generation.py` - Line 66
**Error:** `SyntaxError: closing parenthesis ')' does not match opening parenthesis '{' on line 60`

A syntax error in the test file itself prevents pytest from collecting any tests from this file. The issue appears to be a mismatched bracket/parenthesis in the test code.

---

## Category 2: Anthropic API Tests (8 failures)

All 8 Anthropic integration tests fail with `AssertionError: Result should be a string`. 

**Affected tests:**
- `test_call_valid_response`
- `test_call_error_handling`
- `test_call_json_parsing`
- `test_call_x_api_key_header`
- `test_call_anthropic_version_header`
- `test_call_content_array`
- `test_call_text_extraction`
- `test_call_async`

**Root Cause:** These tests create GrepRAG instances with `external_api_type="openai"` but then call `_call_anthropic()`. The method routing in `GrepRAG._call_external_inference()` checks `external_api_type` and only calls `_call_anthropic()` when it equals `"anthropic"`. Since all instances are configured as OpenAI, the Anthropic call never executes, and the assertions fail.

**Note:** This is a test setup issue - tests should use `external_api_type="anthropic"` when testing Anthropic functionality.

---

## Category 3: Edge Cases Tests (3 failures)

### Test: `test_empty_prompt`
**Error:** `UnboundLocalError: local variable 'queries' referenced before assignment`

**Location:** `tests/test_edge_cases.py`, line ~40-50

The test mocks `_generate_grep_queries()` to return `[".*"]`, then calls `_run_ripgrep()`. However, the error trace suggests the issue is in how `_generate_grep_queries` handles empty prompts. Looking at the implementation, if `prompt.strip() == ""`, it returns an empty list `[]`. But the test's mock may not be properly set up, or there's a logic path where `queries` isn't assigned before use.

### Tests: `test_special_chars_in_prompt`, `test_multiline_strings`
**Error:** Same `UnboundLocalError` pattern

These likely share the same root cause as `test_empty_prompt` - improper handling of edge cases in prompt processing or mocking.

---

## Category 4: Error Handling Tests (5 failures)

All error handling tests fail, suggesting issues in how GrepRAG propagates errors.

### Affected tests:
- `test_missing_ripgrep`
- `test_missing_api_key`
- `test_missing_tree_sitter_parser`
- `test_model_loading_failure`
- `test_blacklist`

**Diagnosis needed:** These tests likely expect specific exception types or error messages that aren't being raised properly. The `_check_dependencies()` method raises a `ValueError` when `external_api_key is None`, but the tests may be passing empty strings instead, bypassing validation. Similar issues could exist for other error paths.

---

## Category 5: AST Ranking Test (1 failure)

### `test_missing_parser_warning`
**Error:** Expected warning not raised or assertion failed

This test verifies that when a tree-sitter parser is missing for a language, GrepRAG logs a warning and falls back to regex scoring. The test likely checks for the presence of a log message, but either:
1. The warning isn't being logged (possibly due to logging level not configured)
2. The assertion about behavior is incorrect

---

## Category 6: Ripgrep Async Test (1 failure)

### `test_parallel_execution`
**Error:** `AssertionError: Result should be a list`  
Actual: `<coroutine object ...mock_run_async>`

**Root Cause:** This is a coroutine handling bug in `_run_ripgrep_async()`. The test mocks `asyncio.to_thread()` to return a coroutine, but the actual implementation isn't properly awaiting it. The method returns a coroutine instead of executing it and returning the result.

The async version likely uses `asyncio.to_thread()` incorrectly, or the test mock setup doesn't match how the function actually awaits results.

---

## Priority Recommendations (for fixing agent)

1. **Immediate:** Fix syntax error in `test_query_generation.py` to enable full test collection.
2. **Critical:** Fix Anthropic tests - configure instances with correct `external_api_type`.
3. **High:** Review edge case handling for empty prompts and special characters.
4. **Medium:** Verify error propagation paths match test expectations.
5. **Low:** Investigate logging/warning behavior in AST ranking, and fix async coroutine issue.
