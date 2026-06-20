import uuid
import json
from typing import Optional, Dict, Any

from .cache import SQLiteCache
from .compressors import compress_code, compress_json, compress_generic_text

class TokenDiet:
    """
    Main orchestrator for compressing context, caching the raw contents,
    and responding to LLM retrieval requests.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.cache = SQLiteCache(db_path)
        
    @property
    def tool_definition(self) -> Dict[str, Any]:
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
                        "description": "The unique ID matching a compressed block placeholder (e.g., ctx_a1b2c3d4)."
                    }
                },
                "required": ["retrieval_id"]
            }
        }

    def compress(self, text: str, threshold: int = 1000, file_extension: Optional[str] = None) -> str:
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
            preview = "\n".join(compressed.splitlines()[:15]) # Keep up to 15 lines of compressed JSON
        elif file_extension or self._is_probable_code(text):
            compressed = compress_code(text, file_extension or ".py")
            preview = "\n".join(compressed.splitlines()[:12]) # Keep up to 12 lines of comments-stripped code
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

    def handle_tool_call(self, tool_input: Dict[str, Any]) -> str:
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
            "import ", "def ", "class ", "function ", "const ", "let ",
            "package ", "struct ", "fn ", "#include", "public class "
        ]
        first_lines = "\n".join(text.splitlines()[:5])
        return any(kw in first_lines for kw in keywords)
