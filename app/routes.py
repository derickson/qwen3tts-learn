import asyncio
import base64
import io
import logging

import anthropic
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
_claude_client = anthropic.Anthropic()


async def translate_text(text: str, target_language: str) -> str:
    """Translate text to the target language using Claude."""
    logger.info("Translating text to %s via Claude API", target_language)
    response = await asyncio.to_thread(
        _claude_client.messages.create,
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Translate the following text to {target_language}. Return ONLY the translation, nothing else:\n\n{text}",
        }],
    )
    return response.content[0].text


class GenerateRequest(BaseModel):
    text: str
    language: str = Field(default="English")
    voice: str = Field(default=config.DEFAULT_VOICE, description="Voice character to use (e.g. 'dave', 'claire')")


class GenerateJsonResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    format: str = "wav"
    text: str


@router.post("/generate", summary="Generate speech audio", response_class=StreamingResponse)
async def generate(req: GenerateRequest, request: Request):
    """Generate speech from text and return a WAV file."""
    if req.voice not in config.VOICES:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unknown voice '{req.voice}'. Available voices: {list(config.VOICES.keys())}"},
        )
    mm = request.app.state.model_manager
    await mm.ensure_loaded()
    text = req.text
    if req.language != "English":
        text = await translate_text(text, req.language)
    wav_bytes, sr = await mm.generate(text, req.language, req.voice)
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="output.wav"'},
    )


@router.post("/generate/json", summary="Generate speech (base64 JSON)", response_model=GenerateJsonResponse)
async def generate_json(req: GenerateRequest, request: Request):
    """Generate speech from text and return base64-encoded audio with metadata."""
    if req.voice not in config.VOICES:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unknown voice '{req.voice}'. Available voices: {list(config.VOICES.keys())}"},
        )
    mm = request.app.state.model_manager
    await mm.ensure_loaded()
    text = req.text
    if req.language != "English":
        text = await translate_text(text, req.language)
    wav_bytes, sr = await mm.generate(text, req.language, req.voice)
    return GenerateJsonResponse(
        audio_base64=base64.b64encode(wav_bytes).decode(),
        sample_rate=sr,
        text=req.text,
    )


@router.get("/getready", summary="Warm up model")
async def get_ready(request: Request):
    """Load the model and create voice prompt if not already loaded."""
    mm = request.app.state.model_manager
    await mm.get_ready()
    return {"status": "ready", "model_loaded": mm.is_loaded}


@router.get("/status", summary="Service status")
async def status(request: Request):
    """Check whether the model is loaded and GPU memory usage."""
    mm = request.app.state.model_manager
    return {
        "model_loaded": mm.is_loaded,
        "gpu": mm.gpu_memory_info(),
    }


@router.post("/unload", summary="Unload model")
async def unload(request: Request):
    """Manually unload the model and free GPU memory."""
    mm = request.app.state.model_manager
    await mm.unload()
    return {"status": "unloaded", "model_loaded": mm.is_loaded}


@router.websocket("/ws/gpu")
async def gpu_websocket(websocket: WebSocket):
    await websocket.accept()
    mm = websocket.app.state.model_manager
    try:
        while True:
            stats = mm.gpu_stats()
            await websocket.send_json(stats)
            interval = 1 if stats["is_generating"] else 15
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
