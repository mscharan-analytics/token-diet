# Token Diet 🥗

[![CI](https://github.com/mscharan-analytics/token-diet/actions/workflows/ci.yml/badge.svg)](https://github.com/mscharan-analytics/token-diet/actions/workflows/ci.yml)

A lightweight, zero-dependency, and extremely easy-to-understand **Context-Compression and Retrieval (CCR)** toolkit for AI agents. 

Reduce prompt sizes by **60% to 90%** on verbose content (like massive tool logs, file listings, and git diffs) while preserving the LLM's ability to recall the original context on demand.

---

## How It Works (The 3-Step Diet)

1. **Compress**: When you pass long text to `token_diet`, it generates a short placeholder preview containing a unique `retrieval_id` (e.g. `ctx_a1b2c3d4`).
2. **Cache**: The full, original text is stored in a local SQLite file (`~/.config/token_diet/cache.db`).
3. **Retrieve**: If the LLM gets confused or needs the exact original content, it invokes a standard tool `retrieve_context(retrieval_id)`. `token_diet` fetches the original text from the local SQLite cache and feeds it back to the LLM.

```
[Agent Tool Output (50,000 tokens)]
             │
             ▼
     ┌──────────────┐
     │  Token Diet  │ ──► [Store full text in local SQLite cache]
     └──────┬───────┘
            │
            ▼
[Compressed Prompt (1,500 tokens)]
            │
            ▼
    ┌──────────────┐
    │  Claude/LLM  │ ──► "Wait, I need the exact details of error ctx_a1b2c3d4."
    └──────┬───────┘
            │  (Calls tool: retrieve_context("ctx_a1b2c3d4"))
            ▼
    ┌──────────────┐
    │  Token Diet  │ ──► [Reads from cache and returns original text]
    └──────────────┘
```

---

## Installation

### 1. Replicate Files directly
Since `token-diet` is designed to be fully transparent, you can simply copy the files in the `token_diet/` folder directly into your own codebase.

### 2. Local Pip Install
Or, you can install it in edit/development mode:
```bash
git clone https://github.com/mscharan-analytics/token-diet.git
cd token-diet
pip install -e .
```

To install with SDK wrappers or local proxy features:
```bash
# For Anthropic/OpenAI SDK wrappers
pip install -e ".[anthropic,openai]"

# For local HTTP proxy features
pip install -e ".[proxy]"
```

---

## Core Usage Example

```python
import anthropic
from token_diet import TokenDiet

# Initialize
diet = TokenDiet()

# 1. Compress a massive string (e.g. build logs or code file)
huge_text = "..." # Imagine a 10,000 line log file
compressed = diet.compress(huge_text, threshold=200)

print(compressed)
# Output:
# [SYSTEM NOTE: The following content was compressed to save tokens.]
# [RETRIEVAL ID: ctx_a1b2c3d4]
# --- PREVIEW ---
# [First few lines of original text...]
# ...
# [To read the complete text, call 'retrieve_context' tool with id='ctx_a1b2c3d4']

# 2. Add it to messages and define the retrieval tool
messages = [{"role": "user", "content": f"Find the issue in this log:\n{compressed}"}]
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=messages,
    tools=[diet.tool_definition] # Register the retrieval tool
)

# 3. Handle retrieval request automatically if Claude calls the tool
if response.stop_reason == "tool_use":
    tool_use = next(block for block in response.content if block.type == "tool_use")
    if tool_use.name == "retrieve_context":
        # Resolve request
        result = diet.handle_tool_call(tool_use.input)
        print("Retrieved original text from cache!")
```

---

## Automatic SDK Client Wrappers (Easiest way)

Instead of manually compressing and handling the retrieval tool call loops yourself, you can wrap your existing **Anthropic** or **OpenAI** client instances. 

The client wrappers automatically intercept your requests, compress long strings, register the retrieval tools, and handle retrieval tool-calls under-the-hood invisibly.

### 1. Anthropic (Claude) Client Wrapper

```python
import anthropic
from token_diet import patch_anthropic_client

# Initialize client as usual
client = anthropic.Anthropic()

# Patch the client instance (default compression threshold is 1000 characters)
client = patch_anthropic_client(client, threshold=1000)

# Call messages.create exactly like you used to do.
# Large inputs will be cached & retrieved automatically behind the scenes!
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": f"Analyze this large codebase file:\n{huge_file_content}"}
    ]
)
print(response.content[0].text)
```

### 2. OpenAI Client Wrapper

```python
import openai
from token_diet import patch_openai_client

client = openai.OpenAI()

# Patch the client
client = patch_openai_client(client, threshold=1000)

# Use OpenAI API as normal
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": f"Parse this database dump:\n{huge_db_dump}"}
    ]
)
print(response.choices[0].message.content)
```

---

## Running the Local HTTP Proxy (For Claude Code, Copilot, Cursor, etc.)

For CLI-based agents like **Claude Code**, **GitHub Copilot CLI**, or **Cursor**, you can run `token-diet` as a local background HTTP proxy. The proxy acts as a middleman between the CLI tool and the upstream LLM API.

### 1. Start the Proxy Server
Once installed with proxy extras, start the server using the shell command:
```bash
token-diet proxy --port 8787
```

### 2. Configure Your Agent Client

#### A. Claude Code
Point Claude Code to the proxy by setting the `ANTHROPIC_BASE_URL` environment variable:
```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787/v1"
claude
```

#### B. GitHub Copilot CLI & OpenAI Clients
Point your client to the local proxy endpoint by modifying the base URL settings:
```bash
export OPENAI_BASE_URL="http://127.0.0.1:8787/v1"
```

---

## Production Readiness: Built-in Failsafes

Token Diet is designed to be fully resilient for production workflows:
1. **Fail-Open Wrap**: If any database, syntax parsing, or connection error happens during the compression or retrieval pipeline, the client wrappers automatically log a warning and fallback to sending the original uncompressed message payload. Your application **never** crashes.
2. **In-Memory Caching Fallback**: If writing context database logs to disk fails (due to write permissions in read-only lambda/server environments), the SQLite cache automatically falls back to a fast, in-memory Python dictionary cache.

---

## Development & Testing
To run the full unit testing suite:
```bash
pip install -e ".[dev]"
pytest
```

---

## Customizing Compression Rules

You can write your own compressors in `token_diet/compressors.py`. By default, the library includes:
- **Code Compressor**: Strips single-line and multi-line comments and minimizes spacing.
- **JSON Crusher**: Drops redundant dict keys or array indexes and keeps structural schemas.
- **Text Compressor**: Truncates prose/text blocks and inserts the retrieval placeholder.

---

## License
MIT License. Feel free to copy, modify, and use in your own projects!
