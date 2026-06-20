import os
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import uvicorn

from .client import TokenDiet

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("token_diet.proxy")

app = FastAPI(title="Token Diet Local Proxy")

# Global TokenDiet instance
diet_instance: Optional[TokenDiet] = None
COMPRESSION_THRESHOLD = 1000

def get_diet() -> TokenDiet:
    global diet_instance
    if diet_instance is None:
        diet_instance = TokenDiet()
    return diet_instance

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """
    Intercepts and optimizes Anthropic's Claude messages.
    URL: /v1/messages
    """
    diet = get_diet()
    body = await request.json()
    messages = body.get("messages", [])
    
    # 1. Compress long messages in place
    messages_copy = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            compressed_content = diet.compress(content, threshold=COMPRESSION_THRESHOLD)
            messages_copy.append({**msg, "content": compressed_content})
        else:
            messages_copy.append(msg)
            
    body["messages"] = messages_copy

    # 2. Inject context retrieval tool
    tools = body.get("tools", [])
    if not any(t.get("name") == "retrieve_context" for t in tools):
        tools_list = list(tools)
        tools_list.append(diet.tool_definition)
        body["tools"] = tools_list

    # Prepare forwarding headers (propagate api-key)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    
    # 3. Request loop to handle CCR resolution
    upstream_url = "https://api.anthropic.com/v1/messages"
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Check if the user is requesting a stream
        is_stream = body.get("stream", False)
        
        while True:
            # If the request requires tool resolution, we must query synchronously first.
            # So even if is_stream was requested, if we stop for tool_use, we resolve it first.
            # Disable stream for intermediate steps
            if is_stream:
                body["stream"] = False
                
            logger.info("Forwarding request to Anthropic upstream...")
            response = await client.post(upstream_url, json=body, headers=headers)
            
            if response.status_code != 200:
                return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))
                
            resp_data = response.json()
            
            # Check if Claude wants to invoke our retrieval tool
            stop_reason = resp_data.get("stop_reason")
            if stop_reason == "tool_use":
                tool_use = next((block for block in resp_data.get("content", []) if block.get("type") == "tool_use"), None)
                if tool_use and tool_use.get("name") == "retrieve_context":
                    retrieval_id = tool_use.get("input", {}).get("retrieval_id")
                    logger.info(f"Intercepted retrieval tool call for: {retrieval_id}")
                    
                    # Fetch from local SQLite cache
                    resolved_text = diet.handle_tool_call(tool_use.get("input", {}))
                    
                    # Append history and loop
                    messages_copy.append({"role": "assistant", "content": resp_data.get("content")})
                    messages_copy.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.get("id"),
                                "content": resolved_text
                            }
                        ]
                    })
                    body["messages"] = messages_copy
                    continue # Keep looping
                    
            # If we reached here, no tool_use was requested, or it's resolved.
            # If the original request wanted a stream, we now re-execute with streaming enabled
            if is_stream:
                body["stream"] = True
                
                # Helper to pipe stream response
                async def stream_generator():
                    async with httpx.AsyncClient(timeout=60.0) as stream_client:
                        async with stream_client.stream("POST", upstream_url, json=body, headers=headers) as r:
                            async for chunk in r.aiter_bytes():
                                yield chunk
                                
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            
            return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """
    Intercepts and optimizes OpenAI's chat completions.
    URL: /v1/chat/completions
    """
    diet = get_diet()
    body = await request.json()
    messages = body.get("messages", [])
    
    # 1. Compress messages in place
    messages_copy = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            compressed_content = diet.compress(content, threshold=COMPRESSION_THRESHOLD)
            messages_copy.append({**msg, "content": compressed_content})
        else:
            messages_copy.append(msg)
            
    body["messages"] = messages_copy

    # 2. Inject tool definition
    tools = body.get("tools", [])
    if not any(t.get("function", {}).get("name") == "retrieve_context" for t in tools):
        tools_list = list(tools)
        tools_list.append(diet.openai_tool_definition)
        body["tools"] = tools_list

    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    upstream_url = "https://api.openai.com/v1/chat/completions"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        is_stream = body.get("stream", False)
        
        while True:
            if is_stream:
                body["stream"] = False
                
            logger.info("Forwarding request to OpenAI upstream...")
            response = await client.post(upstream_url, json=body, headers=headers)
            
            if response.status_code != 200:
                return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))
                
            resp_data = response.json()
            choices = resp_data.get("choices", [])
            if not choices:
                break
                
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            # Check for retrieval tool call
            target_call = next((tc for tc in tool_calls if tc.get("function", {}).get("name") == "retrieve_context"), None)
            if target_call:
                call_args_str = target_call.get("function", {}).get("arguments", "{}")
                try:
                    args_dict = json.loads(call_args_str)
                except ValueError:
                    args_dict = {}
                    
                retrieval_id = args_dict.get("retrieval_id")
                logger.info(f"Intercepted retrieval tool call for: {retrieval_id}")
                
                # Fetch text
                resolved_text = diet.handle_tool_call(args_dict)
                
                # Append assistant tool call and tool result
                messages_copy.append(message)
                messages_copy.append({
                    "role": "tool",
                    "tool_call_id": target_call.get("id"),
                    "name": "retrieve_context",
                    "content": resolved_text
                })
                body["messages"] = messages_copy
                continue
                
            # No tool use or resolved. If user wanted stream, run streaming endpoint
            if is_stream:
                body["stream"] = True
                async def stream_generator():
                    async with httpx.AsyncClient(timeout=60.0) as stream_client:
                        async with stream_client.stream("POST", upstream_url, json=body, headers=headers) as r:
                            async for chunk in r.aiter_bytes():
                                yield chunk
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
                
            return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))


def start_proxy(host: str = "127.0.0.1", port: int = 8787, threshold: int = 1000) -> None:
    """Start the Uvicorn proxy server."""
    global COMPRESSION_THRESHOLD
    COMPRESSION_THRESHOLD = threshold
    logger.info(f"Starting Token Diet Proxy on {host}:{port} (Compression Threshold: {threshold} characters)...")
    uvicorn.run(app, host=host, port=port)
