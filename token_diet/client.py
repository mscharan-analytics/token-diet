import json
import uuid
import warnings
from typing import Any

from .cache import SQLiteCache
from .compressors import compress_code, compress_generic_text, compress_json


class TokenDiet:
    """
    Main orchestrator for compressing context, caching the raw contents,
    and responding to LLM retrieval requests.
    """

    def __init__(self, db_path: str | None = None):
        self.cache = SQLiteCache(db_path)

    @property
    def tool_definition(self) -> dict[str, Any]:
        """
        The tool definition block to supply to Anthropic's Claude SDK.
        """
        return {
            "name": "retrieve_context",
            "description": "Retrieves the full original uncompressed content for a given retrieval ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "retrieval_id": {
                        "type": "string",
                        "description": (
                            "The unique ID matching a compressed block placeholder "
                            "(e.g., ctx_a1b2c3d4)."
                        ),
                    }
                },
                "required": ["retrieval_id"],
            },
        }

    @property
    def openai_tool_definition(self) -> dict[str, Any]:
        """
        The tool definition block to supply to OpenAI's GPT SDK.
        """
        return {
            "type": "function",
            "function": {
                "name": "retrieve_context",
                "description": "Retrieves the full original uncompressed content for a given retrieval ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "retrieval_id": {
                            "type": "string",
                            "description": (
                                "The unique ID matching a compressed block placeholder "
                                "(e.g., ctx_a1b2c3d4)."
                            ),
                        }
                    },
                    "required": ["retrieval_id"],
                },
            },
        }

    def compress(self, text: str, threshold: int = 1000, file_extension: str | None = None) -> str:
        """
        Compresses the text if it is longer than the threshold, stores it in SQLite,
        and returns a placeholder.
        """
        if not text or len(text) <= threshold:
            return text

        # Generate a unique cache key
        cache_id = f"ctx_{uuid.uuid4().hex[:8]}"

        # Cache the original content
        self.cache.put(cache_id, text)

        # Detect the type of content and run compression to generate the preview
        preview = ""
        is_json = False

        # 1. Attempt JSON parsing
        try:
            json.loads(text)
            is_json = True
        except (ValueError, TypeError):
            pass

        if is_json:
            compressed = compress_json(text)
            preview = "\n".join(compressed.splitlines()[:15])  # Keep up to 15 lines of compressed JSON
        elif file_extension or self._is_probable_code(text):
            compressed = compress_code(text, file_extension or ".py")
            preview = "\n".join(compressed.splitlines()[:12])  # Keep up to 12 lines of comments-stripped code
        else:
            compressed = compress_generic_text(text)
            preview = "\n".join(compressed.splitlines()[:8])  # Keep up to 8 lines of text

        # Structure the placeholder prompt
        placeholder = (
            f"[SYSTEM NOTE: The following content was compressed to save tokens.]\n"
            f"[RETRIEVAL ID: {cache_id}]\n"
            f"--- PREVIEW ---\n"
            f"{preview}\n"
            f"...\n"
            f"[To read the complete text, call 'retrieve_context' tool with retrieval_id='{cache_id}']"
        )
        return placeholder

    def handle_tool_call(self, tool_input: dict[str, Any]) -> str:
        """
        Resolves the retrieval tool request.

        Args:
            tool_input: Dictionary matching the tool schema: {"retrieval_id": "ctx_..."}
        """
        retrieval_id = tool_input.get("retrieval_id")
        if not retrieval_id:
            return "Error: Missing retrieval_id parameter."

        original_text = self.cache.get(retrieval_id)
        if original_text is None:
            return f"Error: Context with retrieval ID {retrieval_id} not found."

        return original_text

    def _is_probable_code(self, text: str) -> bool:
        """Heuristic check to guess if a block of text is code."""
        keywords = [
            "import ",
            "def ",
            "class ",
            "function ",
            "const ",
            "let ",
            "package ",
            "struct ",
            "fn ",
            "#include",
            "public class ",
        ]
        first_lines = "\n".join(text.splitlines()[:5])
        return any(kw in first_lines for kw in keywords)


# ==========================================
# SDK WRAPPER HELPERS (FAIL-OPEN IMPLEMENTATIONS)
# ==========================================


def patch_anthropic_client(client: Any, threshold: int = 1000, diet: TokenDiet | None = None) -> Any:
    """
    Patches an Anthropic client instance so that all client.messages.create() calls
    automatically run through the Token Diet compression and retrieval loop.

    Fail-Open: If any exception occurs during compression/wrapper execution,
    it automatically falls back to executing the original uncompressed client call.
    """
    if diet is None:
        diet = TokenDiet()

    original_create = client.messages.create

    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        try:
            messages = kwargs.get("messages")
            if not messages:
                return original_create(*args, **kwargs)

            # Mutating a copy of messages so the developer's original list is untouched
            messages_copy = []
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, str):
                    compressed_content = diet.compress(content, threshold=threshold)
                    messages_copy.append({**msg, "content": compressed_content})
                else:
                    messages_copy.append(msg)

            # Save the modified payload for this call
            kwargs_copy = dict(kwargs)
            kwargs_copy["messages"] = messages_copy

            # Inject retrieval tool definition
            tools = kwargs_copy.get("tools", [])
            if not any(t.get("name") == "retrieve_context" for t in tools):
                tools_list = list(tools)
                tools_list.append(diet.tool_definition)
                kwargs_copy["tools"] = tools_list

            response = original_create(*args, **kwargs_copy)

            # Intercept tool calls automatically in a loop
            while response.stop_reason == "tool_use":
                tool_use = next((block for block in response.content if block.type == "tool_use"), None)
                if not tool_use or tool_use.name != "retrieve_context":
                    break

                # Resolve retrieval ID
                retrieved_text = diet.handle_tool_call(tool_use.input)

                # Append assistant message and tool response back to history
                messages_copy.append({"role": "assistant", "content": response.content})
                messages_copy.append(
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": retrieved_text}],
                    }
                )

                kwargs_copy["messages"] = messages_copy
                response = original_create(*args, **kwargs_copy)

            return response

        except Exception as e:
            # FAIL-OPEN: Log warning and fall back to the raw, unpatched API request
            warnings.warn(
                f"Token Diet encountered an error in wrapped Anthropic client execution: {e}. "
                "Falling back to original uncompressed request.",
                RuntimeWarning, stacklevel=2,
            )
            return original_create(*args, **kwargs)

    client.messages.create = wrapped_create
    return client


def patch_openai_client(client: Any, threshold: int = 1000, diet: TokenDiet | None = None) -> Any:
    """
    Patches an OpenAI client instance so that all client.chat.completions.create() calls
    automatically run through the Token Diet compression and retrieval loop.

    Fail-Open: If any exception occurs during compression/wrapper execution,
    it automatically falls back to executing the original uncompressed client call.
    """
    if diet is None:
        diet = TokenDiet()

    original_create = client.chat.completions.create

    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        try:
            messages = kwargs.get("messages")
            if not messages:
                return original_create(*args, **kwargs)

            # Mutating a copy of messages
            messages_copy = []
            for msg in messages:
                content = msg.get("content")
                if isinstance(content, str):
                    compressed_content = diet.compress(content, threshold=threshold)
                    messages_copy.append({**msg, "content": compressed_content})
                else:
                    messages_copy.append(msg)

            kwargs_copy = dict(kwargs)
            kwargs_copy["messages"] = messages_copy

            # Inject tool definition
            tools = kwargs_copy.get("tools", [])
            if not any(t.get("function", {}).get("name") == "retrieve_context" for t in tools):
                tools_list = list(tools)
                tools_list.append(diet.openai_tool_definition)
                kwargs_copy["tools"] = tools_list

            response = original_create(*args, **kwargs_copy)

            # Loop to automatically resolve tool calls
            while True:
                choice = response.choices[0]
                message = choice.message
                tool_calls = getattr(message, "tool_calls", None)
                if not tool_calls:
                    break

                # Find the retrieve_context tool call
                target_call = next((tc for tc in tool_calls if tc.function.name == "retrieve_context"), None)
                if not target_call:
                    break

                # Parse arguments
                try:
                    args_dict = json.loads(target_call.function.arguments)
                except (ValueError, TypeError):
                    break

                # Fetch uncompressed text
                retrieved_text = diet.handle_tool_call(args_dict)

                # Append history: Assistant's response
                messages_copy.append(message)

                # Append history: Tool resolution result
                messages_copy.append(
                    {
                        "role": "tool",
                        "tool_call_id": target_call.id,
                        "name": "retrieve_context",
                        "content": retrieved_text,
                    }
                )

                kwargs_copy["messages"] = messages_copy
                response = original_create(*args, **kwargs_copy)

            return response

        except Exception as e:
            # FAIL-OPEN: Log warning and fall back to the raw, unpatched API request
            warnings.warn(
                f"Token Diet encountered an error in wrapped OpenAI client execution: {e}. "
                "Falling back to original uncompressed request.",
                RuntimeWarning, stacklevel=2,
            )
            return original_create(*args, **kwargs)

    client.chat.completions.create = wrapped_create
    return client
