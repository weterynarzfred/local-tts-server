import argparse
import re
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHARS = 750
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Text or '-' for stdin")
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--audio-prompt", type=str, default=None)
    args = parser.parse_args()

    text = args.text
    if text == "-":
        text = sys.stdin.read()
    if not text or not text.strip():
        parser.error("--text is required")

    chunks = split_text(text.strip())
    print(f"Chunks: {len(chunks)}", flush=True)

    model = ChatterboxTTS.from_pretrained(device=DEVICE)

    wavs = []
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}/{len(chunks)}: {chunk[:60]}{'...' if len(chunk) > 60 else ''}", flush=True)
        wav = model.generate(
            chunk,
            audio_prompt_path=args.audio_prompt,
            exaggeration=args.exaggeration,
        )
        wavs.append(wav)

    final = stitch(wavs, model.sr) if len(wavs) > 1 else wavs[0]

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = output_dir / f"{timestamp}.wav"
    torchaudio.save(str(wav_path), final, model.sr)

    mp3_path = to_mp3(wav_path)
    print(f"Saved: {mp3_path}")


if __name__ == "__main__":
    main()
