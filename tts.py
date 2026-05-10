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


_TAG_RE = re.compile(r'\[([^\]]+)\]')


def parse_voice_segments(
    text: str,
) -> tuple[list[tuple[str | None, str, str | None]], dict[str, str]]:
    """
    Split tagged text into (voice_name, segment_text, inline_desc) triples.
    inline_desc is set only when the activating tag included a description: [name: desc].
    [name] tags produce inline_desc=None (meaning: use saved sample).
    Also returns voice_descriptions {name: last_desc} as a convenience lookup.
    """
    voice_descriptions: dict[str, str] = {}
    segments: list[tuple[str | None, str, str | None]] = []
    current_voice: str | None = None
    current_inline_desc: str | None = None
    pos = 0

    for m in _TAG_RE.finditer(text):
        before = text[pos:m.start()].strip("\n")
        if before.strip():
            segments.append((current_voice, before, current_inline_desc))

        tag = m.group(1).strip()
        if ":" in tag:
            name, desc = tag.split(":", 1)
            current_voice = name.strip()
            current_inline_desc = desc.strip()
            voice_descriptions[current_voice] = current_inline_desc
        else:
            current_voice = tag.strip()
            current_inline_desc = None

        pos = m.end()

    remaining = text[pos:].strip("\n")
    if remaining.strip():
        segments.append((current_voice, remaining, current_inline_desc))

    if not segments:
        segments = [(None, text, None)]

    return segments, voice_descriptions


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
