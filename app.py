import asyncio
import mimetypes
import re
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from tts import DEVICE, split_text, stitch, to_mp3
from chatterbox.tts import ChatterboxTTS

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
VOICES_DIR = ROOT / "voice_samples"
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg"}

app = FastAPI()

_model: ChatterboxTTS | None = None
_lock = asyncio.Lock()


def _format_label(filename: str) -> str:
    return Path(filename).stem.replace("-", " ").replace("_", " ").title()


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    for src, dst in [("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")]:
        text = text.replace(src, dst)
    return text


def _do_generate(text: str, exaggeration: float, audio_prompt: str | None) -> Path:
    chunks = split_text(text)
    wavs = []
    for chunk in chunks:
        wav = _model.generate(chunk, audio_prompt_path=audio_prompt, exaggeration=exaggeration)
        wavs.append(wav)
    final = stitch(wavs, _model.sr) if len(wavs) > 1 else wavs[0]
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = OUTPUT_DIR / f"{timestamp}.wav"
    torchaudio.save(str(wav_path), final, _model.sr)
    return to_mp3(wav_path)


@app.get("/")
async def index():
    return HTMLResponse((ROOT / "index.html").read_text(encoding="utf-8"))


@app.get("/voices")
async def voices():
    try:
        files = sorted(f for f in VOICES_DIR.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
        return [{"filename": f.name, "label": _format_label(f.name)} for f in files]
    except Exception:
        return []


@app.get("/audio/{filename}")
async def audio(filename: str):
    if Path(filename).name != filename or Path(filename).suffix.lower() not in AUDIO_EXTS:
        raise HTTPException(400, "Invalid filename.")
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found.")
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(file_path, media_type=media_type)


@app.get("/status")
async def status():
    return {"loaded": _model is not None}


@app.post("/unload")
async def unload():
    global _model
    async with _lock:
        if _model is not None:
            del _model
            _model = None
            torch.cuda.empty_cache()
    return {"status": "unloaded"}


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    exaggeration: float = Form(0.5),
    voice: str = Form(""),
):
    global _model

    text = _normalize_text(text)
    if not text:
        raise HTTPException(400, "Text is required.")
    if not 0 <= exaggeration <= 1:
        raise HTTPException(400, "Exaggeration must be between 0 and 1.")

    audio_prompt = None
    if voice:
        fname = Path(voice).name
        if Path(fname).suffix.lower() not in AUDIO_EXTS:
            raise HTTPException(400, "Invalid voice file.")
        voice_path = VOICES_DIR / fname
        if not voice_path.exists():
            raise HTTPException(400, "Voice file not found.")
        audio_prompt = str(voice_path)

    async with _lock:
        if _model is None:
            loop = asyncio.get_running_loop()
            _model = await loop.run_in_executor(None, lambda: ChatterboxTTS.from_pretrained(device=DEVICE))

        loop = asyncio.get_running_loop()
        try:
            mp3_path = await loop.run_in_executor(None, _do_generate, text, exaggeration, audio_prompt)
        except Exception as e:
            raise HTTPException(500, str(e))

    return {"url": f"/audio/{mp3_path.name}", "filename": mp3_path.name}
