import os
import sys
import json
import re

# Append the parent directory to sys.path so we can import token_diet directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from token_diet import patch_anthropic_client, patch_openai_client

# ========================================================
# MOCK CLIENTS FOR KEYLESS SIMULATION
# ========================================================

class MockAnthropicClient:
    """Simulates Anthropic SDK client responses for testing."""
    class Messages:
        def __init__(self):
            self.calls = 0

        def create(self, *args, **kwargs):
            self.calls += 1
            messages = kwargs.get("messages", [])
            
            # Check if there is a placeholder and no tool result yet
            has_placeholder = any("[RETRIEVAL ID: ctx_" in msg.get("content", "") for msg in messages)
            has_tool_result = any(
                isinstance(msg.get("content"), list) and 
                any(
                    (block.get("type") == "tool_result" if isinstance(block, dict) else getattr(block, "type", None) == "tool_result")
                    for block in msg["content"]
                )
                for msg in messages
            )

            class ToolBlock:
                type = "tool_use"
                name = "retrieve_context"
                id = "mock_anthropic_tool_call_id"
                def __init__(self, retrieval_id):
                    self.input = {"retrieval_id": retrieval_id}

            class TextBlock:
                type = "text"
                text = "Mock Anthropic Client: Resolved successfully. I read the uncompressed context and found the segfault on line 1001!"

            class MockResponse:
                def __init__(self, stop_reason, content):
                    self.stop_reason = stop_reason
                    self.content = content

            if has_placeholder and not has_tool_result:
                # Extract retrieval ID from messages to mock LLM calling it
                retrieval_id = "ctx_unknown"
                for msg in messages:
                    match = re.search(r"ctx_[a-f0-9]+", msg.get("content", ""))
                    if match:
                        retrieval_id = match.group(0)
                        break
                print(f"[Mock Anthropic API] Sees placeholder. Stopping with tool_use for: {retrieval_id}")
                return MockResponse("tool_use", [ToolBlock(retrieval_id)])
            else:
                print("[Mock Anthropic API] Sees resolved tool result. Responding with final text.")
                return MockResponse("end_turn", [TextBlock()])

    def __init__(self):
        self.messages = self.Messages()


class MockOpenAIClient:
    """Simulates OpenAI SDK client responses for testing."""
    class ChatCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, *args, **kwargs):
            self.calls += 1
            messages = kwargs.get("messages", [])
            
            # Check if we have a placeholder and no tool responses yet
            has_placeholder = any(
                "[RETRIEVAL ID: ctx_" in (msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "") or "")
                for msg in messages
            )
            has_tool_response = any(
                (msg.get("role") == "tool" if isinstance(msg, dict) else getattr(msg, "role", None) == "tool")
                for msg in messages
            )

            class FunctionCall:
                name = "retrieve_context"
                def __init__(self, retrieval_id):
                    self.arguments = json.dumps({"retrieval_id": retrieval_id})

            class ToolCall:
                id = "mock_openai_tool_call_id"
                type = "function"
                def __init__(self, retrieval_id):
                    self.function = FunctionCall(retrieval_id)

            class MockMessage:
                content = None
                def __init__(self, retrieval_id=None):
                    if retrieval_id:
                        self.tool_calls = [ToolCall(retrieval_id)]
                    else:
                        self.tool_calls = None
                        self.content = "Mock OpenAI Client: Resolved successfully. Found the segfault in uncompressed log!"

            class MockChoice:
                def __init__(self, message):
                    self.message = message

            class MockResponse:
                def __init__(self, choice):
                    self.choices = [choice]

            if has_placeholder and not has_tool_response:
                retrieval_id = "ctx_unknown"
                for msg in messages:
                    match = re.search(r"ctx_[a-f0-9]+", msg.get("content", ""))
                    if match:
                        retrieval_id = match.group(0)
                        break
                print(f"[Mock OpenAI API] Sees placeholder. Stopping with tool_calls for: {retrieval_id}")
                return MockResponse(MockChoice(MockMessage(retrieval_id)))
            else:
                print("[Mock OpenAI API] Sees resolved tool result. Responding with final text.")
                return MockResponse(MockChoice(MockMessage()))

    def __init__(self):
        self.chat = self.ChatCompletions()
        # Mock completions subclass accessor
        self.chat.completions = self.chat


# ========================================================
# MAIN DEMO RUNS
# ========================================================

def run_anthropic_wrapper_demo():
    print("\n=== RUNNING ANTHROPIC CLIENT WRAPPER DEMO ===")
    
    # 1. Initialize our mock client
    raw_client = MockAnthropicClient()
    
    # 2. Patch the client instance with Token Diet
    # We set a low compression threshold of 50 chars so it triggers on our mock payload
    patched_client = patch_anthropic_client(raw_client, threshold=50)
    
    # Huge payload containing log errors
    large_payload = (
        "BOOT LOGS STATUS OK\n"
        "LINE 1: Initializing service...\n"
        "LINE 2: Connected database...\n"
        "LINE 1001: FATAL ERROR: Segment Fault core dumped! Out of RAM memory."
    )
    
    # Send a request like normal.
    # Note: We do NOT define any tool parameters or write manual retrieval loop!
    # The client wrapper handles everything automatically.
    print("Calling patched_client.messages.create()...")
    response = patched_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": f"Find the issue here:\n{large_payload}"}
        ]
    )
    
    print("\nClaude's Final Output:")
    print(response.content[0].text)
    print(f"Total API calls under-the-hood: {raw_client.messages.calls}")


def run_openai_wrapper_demo():
    print("\n=== RUNNING OPENAI CLIENT WRAPPER DEMO ===")
    
    # 1. Initialize mock OpenAI client
    raw_client = MockOpenAIClient()
    
    # 2. Patch client
    patched_client = patch_openai_client(raw_client, threshold=50)
    
    # Large payload
    large_payload = (
        "BOOT LOGS STATUS OK\n"
        "LINE 1001: FATAL ERROR: Segment Fault core dumped! Out of RAM memory."
    )
    
    print("Calling patched_client.chat.completions.create()...")
    response = patched_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"Analyze this log:\n{large_payload}"}
        ]
    )
    
    print("\nGPT's Final Output:")
    print(response.choices[0].message.content)
    print(f"Total API calls under-the-hood: {raw_client.chat.calls}")


if __name__ == "__main__":
    run_anthropic_wrapper_demo()
    run_openai_wrapper_demo()
