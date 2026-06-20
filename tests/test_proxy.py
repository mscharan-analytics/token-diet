import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from token_diet.proxy import app, get_diet

# Initialize the FastAPI TestClient
test_client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_diet_cache():
    """Ensure cache is empty before every test run."""
    diet = get_diet()
    diet.cache.clear()
    yield
    diet.cache.clear()

@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
def test_proxy_anthropic_messages_flow(mock_post):
    """
    Test the Anthropic message proxy route (/v1/messages).
    Verifies that the proxy intercepts prompt text, registers the retrieval tool,
    handles an upstream stop_reason='tool_use' internally to fetch cached context,
    and returns the final resolved response.
    """
    # Mocks for HTTP responses:
    # 1. First upstream call returns tool_use to fetch context
    # 2. Second upstream call returns final completion response
    
    mock_response_1 = MagicMock()
    mock_response_1.status_code = 200
    mock_response_1.json.return_value = {
        "id": "msg_123",
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_xyz",
                "name": "retrieve_context",
                "input": {"retrieval_id": "ctx_temp_id"}  # Will be overridden by actual key during run
            }
        ]
    }
    mock_response_1.content = json.dumps(mock_response_1.json.return_value).encode("utf-8")
    mock_response_1.headers = {}
    
    mock_response_2 = MagicMock()
    mock_response_2.status_code = 200
    mock_response_2.json.return_value = {
        "id": "msg_456",
        "stop_reason": "end_turn",
        "content": [
            {
                "type": "text",
                "text": "Completed! Segment Fault was located and analyzed."
            }
        ]
    }
    mock_response_2.content = json.dumps(mock_response_2.json.return_value).encode("utf-8")
    mock_response_2.headers = {}
    
    mock_post.side_effect = [mock_response_1, mock_response_2]

    # Large log file payload
    huge_log = "\n".join([f"LOG LINE {i}" for i in range(100)]) + "\nFATAL SEGFAULT ERROR!"
    
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": f"Review this crash log:\n{huge_log}"}
        ]
    }

    # Intercept target key of the generated cache:
    # We patch handle_tool_call so we can extract the correct generated ID dynamically during the run
    diet = get_diet()
    original_handle = diet.handle_tool_call
    
    def side_effect_handle(tool_input):
        # Dynamically inject the correct random generated key into the mock response
        retrieval_id = tool_input.get("retrieval_id")
        mock_response_1.json.return_value["content"][0]["input"]["retrieval_id"] = retrieval_id
        mock_response_1.content = json.dumps(mock_response_1.json.return_value).encode("utf-8")
        return original_handle(tool_input)

    with patch.object(diet, "handle_tool_call", side_effect=side_effect_handle):
        response = test_client.post("/v1/messages", json=payload, headers={"x-api-key": "mock-api-key"})

    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["stop_reason"] == "end_turn"
    assert resp_data["content"][0]["text"] == "Completed! Segment Fault was located and analyzed."
    
    # Assert that it made exactly 2 HTTP posts to Anthropic upstream (the CCR loop resolved internally)
    assert mock_post.call_count == 2


@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
def test_proxy_openai_completions_flow(mock_post):
    """
    Test the OpenAI completion proxy route (/v1/chat/completions).
    """
    mock_response_1 = MagicMock()
    mock_response_1.status_code = 200
    mock_response_1.json.return_value = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "retrieve_context",
                                "arguments": json.dumps({"retrieval_id": "ctx_temp_id"})
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    }
    mock_response_1.content = json.dumps(mock_response_1.json.return_value).encode("utf-8")
    mock_response_1.headers = {}
    
    mock_response_2 = MagicMock()
    mock_response_2.status_code = 200
    mock_response_2.json.return_value = {
        "id": "chatcmpl-456",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Completed! Successfully resolved details."
                },
                "finish_reason": "stop"
            }
        ]
    }
    mock_response_2.content = json.dumps(mock_response_2.json.return_value).encode("utf-8")
    mock_response_2.headers = {}
    
    mock_post.side_effect = [mock_response_1, mock_response_2]

    huge_log = "\n".join([f"LOG LINE {i}" for i in range(100)])
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": f"Review this log:\n{huge_log}"}
        ]
    }

    diet = get_diet()
    original_handle = diet.handle_tool_call
    
    def side_effect_handle(tool_input):
        retrieval_id = tool_input.get("retrieval_id")
        mock_response_1.json.return_value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps({"retrieval_id": retrieval_id})
        mock_response_1.content = json.dumps(mock_response_1.json.return_value).encode("utf-8")
        return original_handle(tool_input)

    with patch.object(diet, "handle_tool_call", side_effect=side_effect_handle):
        response = test_client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer mock-key"})

    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["choices"][0]["message"]["content"] == "Completed! Successfully resolved details."
    assert mock_post.call_count == 2
