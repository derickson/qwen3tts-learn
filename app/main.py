import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_VOICE, SUPPORTED_LANGUAGES, VOICES
from .model_manager import ModelManager
from .routes import router as api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    mm = ModelManager()
    app.state.model_manager = mm
    # Auto-load in background so uvicorn finishes binding the port first
    asyncio.create_task(mm.get_ready())
    logger.info("Startup complete — model loading in background.")
    yield
    logger.info("Shutting down — unloading model ...")
    await mm.unload()


app = FastAPI(
    title="Voice Clone TTS",
    description="Text-to-speech API using Qwen3-TTS with Dave's cloned voice. "
                "The model loads on demand and unloads after 15 minutes of inactivity.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing_page(request: Request):
    voice_config = {
        name: {"default_text": v["default_text"], "avatar_video": f"/static/{v['avatar_video']}"}
        for name, v in VOICES.items()
    }
    html = (TEMPLATES_DIR / "index.html").read_text()
    html = html.replace("__VOICE_CONFIG__", json.dumps(voice_config))
    html = html.replace("__DEFAULT_VOICE__", json.dumps(DEFAULT_VOICE))
    html = html.replace("__LANGUAGES__", json.dumps(list(SUPPORTED_LANGUAGES.keys())))
    return HTMLResponse(html)
