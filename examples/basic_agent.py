import os
import sys

# Append the parent directory to sys.path so we can import token_diet directly
# without installing it first. Excellent for people replicating files.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from token_diet import TokenDiet


def run_simulation():
    """
    Simulates a full context compression and retrieval cycle.
    No API keys are required to run this simulation.
    """
    print("=== STARTING TOKEN DIET SIMULATION ===")

    # Initialize the engine
    diet = TokenDiet()

    # Let's create a huge payload (e.g., source code file with comments)
    large_source_code = """
# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
PORT = 8080
DB_URL = "sqlite:///:memory:"

def establish_connection():
    \"\"\"
    Performs a connection handshake with the local database.
    Retries up to 3 times on connection timeout.
    \"\"\"
    # Step 1: Open socket connection
    # Step 2: Perform credential verification
    # Step 3: Return session object
    print("Connecting to DB...")
    return True

# Helper method to calculate mathematical sums
def calculate_sum(a: int, b: int) -> int:
    \"\"\"
    Sums two integers and returns the result.
    \"\"\"
    # Basic addition
    return a + b
"""

    print(f"Original Text Length: {len(large_source_code)} characters")

    # Compress the code using a low threshold of 100 characters so it triggers
    compressed_result = diet.compress(large_source_code, threshold=100, file_extension=".py")

    print("\n--- Compressed Prompt (Sent to LLM) ---")
    print(compressed_result)

    # Parse out the retrieval ID from the text to simulate the LLM's action
    import re

    match = re.search(r"ctx_[a-f0-9]+", compressed_result)
    if not match:
        print("Error: Could not find retrieval ID in placeholder.")
        return

    retrieval_id = match.group(0)
    print("\n--- Simulated LLM Tool Call ---")
    print(f"LLM calls retrieve_context(retrieval_id='{retrieval_id}')")

    # Simulate resolving the tool call
    resolved_text = diet.handle_tool_call({"retrieval_id": retrieval_id})

    print("\n--- Retrieved Original Content ---")
    print(resolved_text)

    assert resolved_text == large_source_code
    print("\nSimulation succeeded! The retrieved context is byte-for-byte identical to the original.")


def run_live_anthropic():
    """Runs a live context compression request using the Anthropic SDK."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[Skip Live Test] ANTHROPIC_API_KEY not found in environment.")
        print("To run a live test, execute: export ANTHROPIC_API_KEY='your-key' && python examples/basic_agent.py")
        return

    try:
        import anthropic
    except ImportError:
        print("\n[Skip Live Test] 'anthropic' package not installed.")
        print("Install optional dependencies: pip install -e '.[anthropic]'")
        return

    print("\n=== STARTING LIVE ANTHROPIC CLAUDE RUN ===")

    diet = TokenDiet()
    client = anthropic.Anthropic(api_key=api_key)

    # Build a huge dummy context
    huge_log = "\n".join([f"LINE {i}: system initialized successfully, status code: 200" for i in range(1000)])
    huge_log += "\nLINE 1001: FATAL ERROR: Segment Fault core dumped! Out of RAM memory."

    # Compress it
    compressed_log = diet.compress(huge_log, threshold=300)

    messages = [
        {
            "role": "user",
            "content": (
                "Please examine this system boot log and tell me "
                f"what the final error code is:\n\n{compressed_log}"
            ),
        }
    ]

    print("Sending prompt to Claude...")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022", max_tokens=1000, messages=messages, tools=[diet.tool_definition]
    )

    print(f"Claude Response Status: stop_reason={response.stop_reason}")

    # Process potential tool calls
    if response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")
        print(f"Claude invoked tool '{tool_use.name}' with input: {tool_use.input}")

        # Resolve the tool result
        tool_result = diet.handle_tool_call(tool_use.input)

        # Append the tool call and response back to the history
        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": tool_result}]}
        )

        # Resubmit to Claude to get the final answer
        print("Submitting resolved tool result back to Claude...")
        final_response = client.messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=1000, messages=messages, tools=[diet.tool_definition]
        )

        print("\n--- Claude's Final Answer ---")
        print(final_response.content[0].text)
    else:
        print("\n--- Claude's Response ---")
        print(response.content[0].text)


if __name__ == "__main__":
    run_simulation()
    run_live_anthropic()
