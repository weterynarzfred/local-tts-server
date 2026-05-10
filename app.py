import asyncio
import json
import mimetypes
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from tts import DEVICE, split_text, stitch, to_mp3, parse_voice_segments
from chatterbox.tts import ChatterboxTTS

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
VOICES_DIR = ROOT / "voice_samples"
KOKORO_ROOT = Path("H:/tts/kokoro")
KOKORO_MODEL_ROOT = KOKORO_ROOT / "models" / "Kokoro-82M"
KOKORO_VOICES_DIR = KOKORO_MODEL_ROOT / "voices"
KOKORO_SR = 24000
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg"}

QWEN3_VD_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN3_BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN3_VOICES_DIR = ROOT / "voice_samples" / "qwen3"
QWEN3_LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]
QWEN3_DEFAULT_VOICE_DESC = "Warm female narrator, slight British accent, calm pace"
QWEN3_REF_TEXT = (
    "The morning light filtered through the curtains, casting soft patterns on the wooden floor. "
    "She picked up her book and settled into the armchair by the window, "
    "ready to lose herself in another story for a while."
)

KOKORO_LANG_CODES = {
    "a": {"af", "am"},
    "b": {"bf", "bm"},
    "e": {"ef", "em"},
    "f": {"ff"},
    "h": {"hf", "hm"},
    "i": {"if", "im"},
    "j": {"jf", "jm"},
    "p": {"pf", "pm"},
    "z": {"zf", "zm"},
}
_PREFIX_TO_LANG = {prefix: code for code,
                   prefixes in KOKORO_LANG_CODES.items() for prefix in prefixes}

app = FastAPI()

_chatterbox_model: ChatterboxTTS | None = None
_kokoro_kmodel = None
_qwen3_model: Any = None
_qwen3_base_model: Any = None
_lock = asyncio.Lock()


def _resolve_chatterbox_voice(voice_name: str | None, default: str | None) -> str | None:
  if voice_name is None:
    return default
  for ext in AUDIO_EXTS:
    p = VOICES_DIR / f"{voice_name}{ext}"
    if p.exists():
      return str(p)
  return default


def _resolve_kokoro_voice(voice_name: str | None, default_stem: str) -> str:
  stem = voice_name if voice_name else default_stem
  p = KOKORO_VOICES_DIR / f"{stem}.pt"
  if p.exists():
    return str(p)
  return str(KOKORO_VOICES_DIR / f"{default_stem}.pt")


def _format_label(filename: str) -> str:
  return Path(filename).stem.replace("-", " ").replace("_", " ").title()


def _normalize_text(text: str) -> str:
  text = text.strip()
  text = re.sub(r"[^\S\n]+", " ", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  for src, dst in [("'", "'"), ("'", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")]:
    text = text.replace(src, dst)
  return text


def _kokoro_lang(voice_stem: str) -> str:
  return _PREFIX_TO_LANG.get(voice_stem[:2], "a")


def _sse(event: str, data: dict) -> str:
  return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- Chatterbox generation ---

def _do_chatterbox_generate(text: str, exaggeration: float, default_audio_prompt: str | None,
                            progress_q: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> Path:
  import tqdm as tqdm_module

  segments, _ = parse_voice_segments(text)
  all_chunks: list[tuple[str | None, str]] = []
  for voice_name, seg_text, _ in segments:
    ap = _resolve_chatterbox_voice(voice_name, default_audio_prompt)
    for chunk in split_text(seg_text):
      all_chunks.append((ap, chunk))

  n_chunks = len(all_chunks)
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
    for i, (ap, chunk) in enumerate(all_chunks):
      current_chunk[0] = i
      wav = _chatterbox_model.generate(
        chunk, audio_prompt_path=ap, exaggeration=exaggeration)
      wavs.append(wav)
    final = stitch(wavs, _chatterbox_model.sr) if len(wavs) > 1 else wavs[0]
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = OUTPUT_DIR / f"{timestamp}.wav"
    torchaudio.save(str(wav_path), final, _chatterbox_model.sr)
    return to_mp3(wav_path)
  finally:
    tqdm_module.tqdm.update = original_update
    loop.call_soon_threadsafe(progress_q.put_nowait, None)


# --- Kokoro generation ---

def _load_kokoro_kmodel():
  from kokoro.model import KModel
  return (
      KModel(
          repo_id="hexgrad/Kokoro-82M",
          config=str(KOKORO_MODEL_ROOT / "config.json"),
          model=str(KOKORO_MODEL_ROOT / "kokoro-v1_0.pth"),
      )
      .to(DEVICE)
      .eval()
  )


def _do_kokoro_generate(text: str, speed: float, default_voice_path: str,
                        progress_q: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> Path:
  from kokoro import KPipeline

  segments, _ = parse_voice_segments(text)
  default_stem = Path(default_voice_path).stem

  audio_parts = []
  for voice_name, seg_text, _ in segments:
    voice_path = _resolve_kokoro_voice(voice_name, default_stem)
    voice_stem = Path(voice_path).stem
    lang_code = _kokoro_lang(voice_stem)
    pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M",
                         model=_kokoro_kmodel, device=DEVICE)
    for result in pipeline(seg_text, voice=voice_path, speed=speed):
      audio = result.audio if hasattr(result, "audio") else result[2]
      audio_parts.append(np.asarray(audio, dtype=np.float32))

  if not audio_parts:
    raise RuntimeError("No audio generated.")

  final_np = np.concatenate(audio_parts)
  wav_tensor = torch.from_numpy(final_np).unsqueeze(0)
  OUTPUT_DIR.mkdir(exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  wav_path = OUTPUT_DIR / f"{timestamp}.wav"
  torchaudio.save(str(wav_path), wav_tensor, KOKORO_SR)
  result_path = to_mp3(wav_path)
  loop.call_soon_threadsafe(progress_q.put_nowait, None)
  return result_path


# --- Qwen3 VoiceDesign generation ---

def _load_qwen3_model():
  from qwen_tts import Qwen3TTSModel
  return Qwen3TTSModel.from_pretrained(
      QWEN3_VD_MODEL_ID,
      device_map="cuda:0" if torch.cuda.is_available() else "cpu",
      dtype=torch.bfloat16,
      attn_implementation="sdpa",
  )


def _load_qwen3_base_model():
  from qwen_tts import Qwen3TTSModel
  return Qwen3TTSModel.from_pretrained(
      QWEN3_BASE_MODEL_ID,
      device_map="cuda:0" if torch.cuda.is_available() else "cpu",
      dtype=torch.bfloat16,
      attn_implementation="sdpa",
  )


def _qwen3_vd_generate(seg_text: str, language: str, instruct: str) -> tuple[np.ndarray, int]:
  wavs, sr = _qwen3_model.generate_voice_design(text=seg_text, language=language, instruct=instruct)
  wav_np = wavs[0] if isinstance(wavs, list) else wavs
  return np.asarray(wav_np, dtype=np.float32), sr


def _qwen3_clone_generate(seg_text: str, language: str, ref_wav_path: Path) -> tuple[np.ndarray, int]:
  ref_txt_path = ref_wav_path.with_suffix(".txt")
  ref_text = ref_txt_path.read_text(encoding="utf-8") if ref_txt_path.exists() else None
  wavs, sr = _qwen3_base_model.generate_voice_clone(
      text=seg_text,
      language=language,
      ref_audio=str(ref_wav_path),
      ref_text=ref_text,
      x_vector_only_mode=(ref_text is None),
  )
  wav_np = wavs[0] if isinstance(wavs, list) else wavs
  return np.asarray(wav_np, dtype=np.float32), sr


def _do_qwen3_generate(text: str, language: str, default_voice_description: str,
                       progress_q: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> Path:
  segments, _ = parse_voice_segments(text)
  QWEN3_VOICES_DIR.mkdir(parents=True, exist_ok=True)

  audio_parts = []
  final_sr = None

  for voice_name, seg_text, inline_desc in segments:
    seg_text = seg_text.strip()
    if not seg_text:
      continue

    if voice_name is None:
      # Untagged text: VoiceDesign with default description
      wav_np, sr = _qwen3_vd_generate(seg_text, language, default_voice_description)
    elif inline_desc is not None:
      # [name: description] — generate reference sample, save it, then clone
      ref_wav_path = QWEN3_VOICES_DIR / f"{voice_name}.wav"
      ref_txt_path = ref_wav_path.with_suffix(".txt")
      ref_np, ref_sr = _qwen3_vd_generate(QWEN3_REF_TEXT, language, inline_desc)
      ref_tensor = torch.from_numpy(ref_np).unsqueeze(0)
      torchaudio.save(str(ref_wav_path), ref_tensor, ref_sr)
      ref_txt_path.write_text(QWEN3_REF_TEXT, encoding="utf-8")
      wav_np, sr = _qwen3_clone_generate(seg_text, language, ref_wav_path)
    else:
      # [name] — use saved reference sample for cloning
      ref_wav_path = QWEN3_VOICES_DIR / f"{voice_name}.wav"
      if ref_wav_path.exists():
        wav_np, sr = _qwen3_clone_generate(seg_text, language, ref_wav_path)
      else:
        # No saved sample: fall back to VoiceDesign with default
        wav_np, sr = _qwen3_vd_generate(seg_text, language, default_voice_description)

    final_sr = sr
    audio_parts.append(wav_np)

  if not audio_parts:
    raise RuntimeError("No audio generated.")

  final_np = np.concatenate(audio_parts)
  wav_tensor = torch.from_numpy(final_np).unsqueeze(0)
  OUTPUT_DIR.mkdir(exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  wav_path = OUTPUT_DIR / f"{timestamp}.wav"
  torchaudio.save(str(wav_path), wav_tensor, final_sr)
  result_path = to_mp3(wav_path)
  loop.call_soon_threadsafe(progress_q.put_nowait, None)
  return result_path


# --- Routes ---

@app.get("/")
async def index():
  return HTMLResponse((ROOT / "index.html").read_text(encoding="utf-8"))


@app.get("/voices")
async def voices():
  chatterbox = []
  try:
    files = sorted(f for f in VOICES_DIR.iterdir() if f.is_file()
                   and f.suffix.lower() in AUDIO_EXTS)
    chatterbox = [{"filename": f.name,
                   "label": _format_label(f.name)} for f in files]
  except Exception:
    pass

  kokoro = []
  try:
    files = sorted(f for f in KOKORO_VOICES_DIR.iterdir() if f.suffix == ".pt")
    kokoro = [{"filename": f.name,
               "label": _format_label(f.name)} for f in files]
  except Exception:
    pass

  return {"chatterbox": chatterbox, "kokoro": kokoro, "qwen3": []}


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
  return {
      "chatterbox": _chatterbox_model is not None,
      "kokoro": _kokoro_kmodel is not None,
      "qwen3_vd": _qwen3_model is not None,
      "qwen3_base": _qwen3_base_model is not None,
  }


@app.post("/unload")
async def unload(engine: str = Form("")):
  global _chatterbox_model, _kokoro_kmodel, _qwen3_model, _qwen3_base_model
  async with _lock:
    if engine in ("chatterbox", ""):
      if _chatterbox_model is not None:
        del _chatterbox_model
        _chatterbox_model = None
    if engine in ("kokoro", ""):
      if _kokoro_kmodel is not None:
        del _kokoro_kmodel
        _kokoro_kmodel = None
    if engine in ("qwen3", ""):
      if _qwen3_model is not None:
        del _qwen3_model
        _qwen3_model = None
      if _qwen3_base_model is not None:
        del _qwen3_base_model
        _qwen3_base_model = None
    torch.cuda.empty_cache()
  return {"status": "unloaded"}


@app.post("/synthesize")
async def synthesize(
    engine: str = Form("chatterbox"),
    text: str = Form(...),
    exaggeration: float = Form(0.5),
    speed: float = Form(1.0),
    voice: str = Form(""),
    voice_description: str = Form(""),
    language: str = Form("English"),
):
  global _chatterbox_model, _kokoro_kmodel, _qwen3_model, _qwen3_base_model

  text = _normalize_text(text)
  if not text:
    raise HTTPException(400, "Text is required.")

  loop = asyncio.get_running_loop()
  progress_q: asyncio.Queue = asyncio.Queue()

  if engine == "chatterbox":
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

    async def stream():
      global _chatterbox_model
      async with _lock:
        if _chatterbox_model is None:
          try:
            _chatterbox_model = await loop.run_in_executor(
                None, lambda: ChatterboxTTS.from_pretrained(device=DEVICE)
            )
          except Exception as e:
            yield _sse("error", {"error": str(e)})
            return

        future = loop.run_in_executor(
            None, _do_chatterbox_generate, text, exaggeration, audio_prompt, progress_q, loop
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

  elif engine == "kokoro":
    if not 0.5 <= speed <= 2.0:
      raise HTTPException(400, "Speed must be between 0.5 and 2.")

    voice_stem = Path(voice).stem if voice else "af_heart"
    voice_path = str(KOKORO_VOICES_DIR / f"{voice_stem}.pt")
    if not Path(voice_path).exists():
      raise HTTPException(400, "Voice file not found.")

    async def stream():
      global _kokoro_kmodel
      async with _lock:
        if _kokoro_kmodel is None:
          try:
            _kokoro_kmodel = await loop.run_in_executor(None, _load_kokoro_kmodel)
          except Exception as e:
            yield _sse("error", {"error": str(e)})
            return

        future = loop.run_in_executor(
            None, _do_kokoro_generate, text, speed, voice_path, progress_q, loop
        )
        await progress_q.get()  # wait for None sentinel
        try:
          mp3_path = await future
          yield _sse("done", {"url": f"/audio/{mp3_path.name}", "filename": mp3_path.name})
        except Exception as e:
          yield _sse("error", {"error": str(e)})

  elif engine == "qwen3":
    if language not in QWEN3_LANGUAGES:
      raise HTTPException(400, f"Unsupported language: {language}")
    effective_desc = voice_description.strip() or QWEN3_DEFAULT_VOICE_DESC

    async def stream():
      global _qwen3_model, _qwen3_base_model
      async with _lock:
        if _qwen3_model is None:
          try:
            _qwen3_model = await loop.run_in_executor(None, _load_qwen3_model)
          except Exception as e:
            yield _sse("error", {"error": str(e)})
            return
        if _qwen3_base_model is None:
          try:
            _qwen3_base_model = await loop.run_in_executor(None, _load_qwen3_base_model)
          except Exception as e:
            yield _sse("error", {"error": str(e)})
            return

        future = loop.run_in_executor(
            None, _do_qwen3_generate, text, language, effective_desc, progress_q, loop
        )
        await progress_q.get()
        try:
          mp3_path = await future
          yield _sse("done", {"url": f"/audio/{mp3_path.name}", "filename": mp3_path.name})
        except Exception as e:
          yield _sse("error", {"error": str(e)})

  else:
    raise HTTPException(400, f"Unknown engine: {engine}")

  return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
