import asyncio
import json
import mimetypes
import re
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

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
    for src, dst in [("'", "'"), ("'", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")]:
        text = text.replace(src, dst)
    return text


def _do_generate_with_progress(text: str, exaggeration: float, audio_prompt: str | None,
                                progress_q: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> Path:
    import tqdm as tqdm_module

    chunks = split_text(text)
    n_chunks = len(chunks)
    current_chunk = [0]

    original_update = tqdm_module.tqdm.update

    def patched_update(self, n=1):
        result = original_update(self, n)
        if self.total:
            overall = (current_chunk[0] + self.n / self.total) / n_chunks
            loop.call_soon_threadsafe(progress_q.put_nowait, min(overall, 1.0))
        return result

    tqdm_module.tqdm.update = patched_update
    try:
        wavs = []
        for i, chunk in enumerate(chunks):
            current_chunk[0] = i
            wav = _model.generate(chunk, audio_prompt_path=audio_prompt, exaggeration=exaggeration)
            wavs.append(wav)

        final = stitch(wavs, _model.sr) if len(wavs) > 1 else wavs[0]
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = OUTPUT_DIR / f"{timestamp}.wav"
        torchaudio.save(str(wav_path), final, _model.sr)
        return to_mp3(wav_path)
    finally:
        tqdm_module.tqdm.update = original_update
        loop.call_soon_threadsafe(progress_q.put_nowait, None)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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

    loop = asyncio.get_running_loop()
    progress_q: asyncio.Queue = asyncio.Queue()

    async def stream():
        global _model
        async with _lock:
            if _model is None:
                try:
                    _model = await loop.run_in_executor(None, lambda: ChatterboxTTS.from_pretrained(device=DEVICE))
                except Exception as e:
                    yield _sse("error", {"error": str(e)})
                    return

            future = loop.run_in_executor(
                None, _do_generate_with_progress, text, exaggeration, audio_prompt, progress_q, loop
            )

            while True:
                value = await progress_q.get()
                if value is None:
                    break
                yield _sse("progress", {"value": value})

            try:
                mp3_path = await future
                yield _sse("done", {"url": f"/audio/{mp3_path.name}", "filename": mp3_path.name})
            except Exception as e:
                yield _sse("error", {"error": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
