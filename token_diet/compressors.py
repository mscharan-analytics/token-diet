import re
import json
from typing import Any

def compress_code(code: str, file_extension: str = ".py") -> str:
    """
    Remove comments, docstrings, and collapse excessive spacing
    to reduce token size without changing code behavior.
    """
    # Normalize newlines
    code = code.replace("\r\n", "\n")
    
    # 1. Remove comments depending on language
    if file_extension in (".py", ".yaml", ".yml", ".ini"):
        # Remove Python comments (lines starting with #, ignoring # inside quotes is hard without AST,
        # but a simple regex covers most user scripts & tool prints)
        code = re.sub(r'(?m)^[ \t]*#[^\n]*\n?', '', code)
        code = re.sub(r'[ \t]+#[^\n]*', '', code)
        
        # Remove multi-line docstrings (''' ... ''' or \"\"\" ... \"\"\")
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        code = re.sub(r"'''[\s\S]*?'''", '', code)
        
    elif file_extension in (".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".cpp", ".c", ".h"):
        # Remove JS/TS/Go multi-line comments /* ... */
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        # Remove single line comments //
        code = re.sub(r'(?m)^[ \t]*//[^\n]*\n?', '', code)
        code = re.sub(r'[ \t]+//[^\n]*', '', code)

    # 2. Collapse multiple consecutive empty lines to a single newline
    code = re.sub(r'\n\s*\n', '\n', code)
    
    return code.strip()

def compress_json(json_str: str) -> str:
    """
    Parses a JSON string and collapses large arrays of dicts into a summary format,
    preserving structural schema and representative samples.
    """
    try:
        data = json.loads(json_str)
        compressed_data = _compress_json_value(data)
        return json.dumps(compressed_data, indent=2)
    except (json.JSONDecodeError, TypeError):
        return json_str

def _compress_json_value(val: Any) -> Any:
    """Recursively processes and compresses JSON values."""
    if isinstance(val, list):
        if not val:
            return val
        
        # If it's a huge array of dictionaries (e.g. database query rows)
        if len(val) > 3 and isinstance(val[0], dict):
            # Take a sample of the first 2 items
            sample = [_compress_json_value(item) for item in val[:2]]
            keys = list(val[0].keys())
            summary = {
                "__token_diet_summary__": f"Array truncated from {len(val)} items to 2 sample items.",
                "all_keys_in_objects": keys,
                "samples": sample
            }
            return summary
        
        # Otherwise, process list items recursively
        return [_compress_json_value(item) for item in val]
        
    elif isinstance(val, dict):
        # Recursively compress dictionary values
        return {k: _compress_json_value(v) for k, v in val.items()}
        
    elif isinstance(val, str) and len(val) > 2000:
        # Truncate overly long strings inside JSON fields
        return val[:200] + f" ... [TRUNCATED {len(val) - 400} CHARS] ... " + val[-200:]
        
    return val

def compress_generic_text(text: str) -> str:
    """Collapses consecutive multiple empty lines and excessive whitespace."""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
