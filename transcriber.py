"""
transcriber.py

Audio transcription engine using OpenAI Whisper.
Handles M4A, CAF, WAV, and MP3 files with segment-level timestamps.
"""

import shutil
import sys
from pathlib import Path

import whisper

SUPPORTED_EXTENSIONS = {".m4a", ".caf", ".wav", ".mp3"}

MODEL_SIZES = ("tiny", "base", "small", "medium", "large")


def check_ffmpeg():
    """Raise RuntimeError if ffmpeg is not installed."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH.\n"
            "  macOS:  brew install ffmpeg\n"
            "  Linux:  sudo apt install ffmpeg"
        )


def load_model(model_size: str = "base") -> whisper.Whisper:
    """Load a Whisper model by size name."""
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size '{model_size}'. Choose from {MODEL_SIZES}")
    check_ffmpeg()
    return whisper.load_model(model_size)


def transcribe(model: whisper.Whisper, audio_path: str | Path) -> dict:
    """
    Transcribe an audio file and return a structured result.

    Returns:
        {
            "source_file": str,
            "language": str,
            "text": str,             # full transcript
            "segments": [
                {"id": int, "start": float, "end": float, "text": str}, ...
            ]
        }
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{audio_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    raw = model.transcribe(str(audio_path))

    segments = [
        {
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        for seg in raw.get("segments", [])
    ]

    return {
        "source_file": str(audio_path),
        "language": raw.get("language", "unknown"),
        "text": raw["text"].strip(),
        "segments": segments,
    }


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    total_s = int(round(seconds))
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
