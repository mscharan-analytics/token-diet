from token_diet.compressors import compress_code, compress_json, compress_generic_text
import json

def test_compress_code_python():
    code = """# Some comment
def hello():
    \"\"\"This is a docstring.\"\"\"
    # Inline comment
    print("Hello world")  # Prints string
"""
    result = compress_code(code, ".py")
    assert "Some comment" not in result
    assert "This is a docstring" not in result
    assert "Inline comment" not in result
    assert "print(\"Hello world\")" in result

def test_compress_code_javascript():
    code = """// JavaScript test
function add(a, b) {
    /* Multi-line
       comment */
    return a + b; // inline sum
}
"""
    result = compress_code(code, ".js")
    assert "JavaScript test" not in result
    assert "Multi-line" not in result
    assert "inline sum" not in result
    assert "return a + b;" in result

def test_compress_json_arrays():
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"},
        {"id": 4, "name": "David"}
    ]
    json_str = json.dumps(data)
    result_str = compress_json(json_str)
    result = json.loads(result_str)
    
    assert "__token_diet_summary__" in result
    assert "Array truncated" in result["__token_diet_summary__"]
    assert len(result["samples"]) == 2
    assert result["all_keys_in_objects"] == ["id", "name"]

def test_compress_generic_text():
    text = "Hello    world.\n\n\nThis is\t\t a test."
    result = compress_generic_text(text)
    assert "Hello world." in result
    # Multiple tabs/spaces collapsed to single space
    assert "This is a test." in result
