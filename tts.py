import re
import subprocess
import warnings
from pathlib import Path

import torch
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHARS = 500
SILENCE_S = 0.3

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def split_text(text: str) -> list[str]:
    pieces = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= MAX_CHARS:
            pieces.append(para)
        else:
            pieces.extend(_split_long(para))
    return _greedy_join(pieces)


def _split_long(text: str) -> list[str]:
    chunks = _greedy_join(re.split(r"(?<=\.)\s+", text))
    result = []
    for chunk in chunks:
        if len(chunk) > MAX_CHARS:
            result.extend(_greedy_join(re.split(r"(?<=,)\s+", chunk)))
        else:
            result.append(chunk)
    return result


def _greedy_join(parts: list[str]) -> list[str]:
    chunks, current = [], ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= MAX_CHARS:
            current += " " + part
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


def stitch(wavs: list[torch.Tensor], sr: int) -> torch.Tensor:
    normalized = [w.unsqueeze(0) if w.dim() == 1 else w for w in wavs]
    silence = torch.zeros_like(normalized[0][:, : int(SILENCE_S * sr)])
    parts = []
    for i, wav in enumerate(normalized):
        parts.append(wav)
        if i < len(normalized) - 1:
            parts.append(silence)
    return torch.cat(parts, dim=1)


def to_mp3(wav_path: Path) -> Path:
    mp3_path = wav_path.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-qscale:a", "2", str(mp3_path)],
        check=True,
        capture_output=True,
    )
    wav_path.unlink()
    return mp3_path
