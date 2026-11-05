"""Keyless transport test only: never executes actions or claims a task was saved."""
import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    text = "This is a local voice pipeline test. No task was created."
    if body.get("stream"):
        async def chunks():
            chunk = {"id": "local-test", "object": "chat.completion.chunk",
                     "model": "local-test-stub", "choices": [{"index": 0,
                     "delta": {"content": text}, "finish_reason": None}]}
            yield "data: " + json.dumps(chunk) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(chunks(), media_type="text/event-stream")
    return {"id": "local-test", "object": "chat.completion", "model": "local-test-stub",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}]}
