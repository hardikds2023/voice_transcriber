#!/usr/bin/env python3
"""
voice_memo_transcriber.py

Transcribe Apple Voice Memos (and any .m4a/.caf files) using OpenAI Whisper.

Usage:
    python voice_memo_transcriber.py [--input PATH] [--model SIZE]
                                     [--output DIR] [--format {txt,json,srt}]

Examples:
    # Auto-detect macOS Voice Memos folder and transcribe everything
    python voice_memo_transcriber.py

    # Transcribe a single file and save as SRT subtitles
    python voice_memo_transcriber.py --input memo.m4a --format srt

    # Transcribe a directory with a more accurate model, output JSON
    python voice_memo_transcriber.py --input ~/recordings --model small --format json

    # Save all transcripts to a dedicated folder
    python voice_memo_transcriber.py --output ~/transcripts

Requirements:
    pip install openai-whisper
    brew install ffmpeg   # (macOS)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import whisper


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".m4a", ".caf"}

# macOS paths tried in order from newest to oldest OS version
VOICE_MEMOS_CANDIDATE_PATHS = [
    "~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings",
    "~/Library/Containers/com.apple.VoiceMemos/Data/Library/Application Support",
    "~/Library/Application Support/com.apple.voicememos/Recordings",
    "~/Music/iTunes/iTunes Media/Voice Memos",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe Apple Voice Memos using OpenAI Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python voice_memo_transcriber.py
  python voice_memo_transcriber.py --input ~/recordings --model small
  python voice_memo_transcriber.py --input memo.m4a --format srt
  python voice_memo_transcriber.py --output ~/transcripts --format json
        """,
    )
    parser.add_argument(
        "--input",
        metavar="PATH",
        help=(
            "Path to a single .m4a/.caf file or a directory of audio files. "
            "If omitted, the macOS Voice Memos folder is auto-detected."
        ),
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base). Larger = slower but more accurate.",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        help=(
            "Directory to write transcription files. "
            "Defaults to the same directory as each audio file."
        ),
    )
    parser.add_argument(
        "--format",
        default="txt",
        choices=["txt", "json", "srt"],
        help="Output file format: plain text (default), JSON with segments, or SRT subtitles.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def detect_voice_memos_dir():
    """Return the first existing macOS Voice Memos directory, or None."""
    for candidate in VOICE_MEMOS_CANDIDATE_PATHS:
        path = Path(candidate).expanduser()
        if path.is_dir():
            return path
    return None


def resolve_input_path(args):
    """Return a sorted list of audio file Paths to process."""
    if args.input:
        p = Path(args.input).expanduser()
        if not p.exists():
            print(f"ERROR: Input path not found: {p}", file=sys.stderr)
            sys.exit(1)
        if p.is_file():
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                print(
                    f"ERROR: Unsupported file type '{p.suffix}'. Expected .m4a or .caf.",
                    file=sys.stderr,
                )
                sys.exit(1)
            return [p]
        # Directory
        files = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print(f"ERROR: No .m4a or .caf files found in {p}", file=sys.stderr)
            sys.exit(1)
        return files

    # Auto-detect
    voice_memos_dir = detect_voice_memos_dir()
    if voice_memos_dir is None:
        print(
            "ERROR: Could not find the Voice Memos directory.\n"
            "       Pass --input to specify a path, or ensure Voice Memos are\n"
            "       stored locally (disable iCloud-only sync if needed).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Auto-detected Voice Memos folder: {voice_memos_dir}")
    files = sorted(
        f for f in voice_memos_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"ERROR: No .m4a or .caf files found in {voice_memos_dir}", file=sys.stderr)
        sys.exit(1)
    return files


def get_output_path(audio_path, args):
    """Compute the output file path for a given audio file."""
    ext = f".{args.format}"
    output_dir = Path(args.output).expanduser() if args.output else audio_path.parent
    return output_dir / (audio_path.stem + ext)


def already_transcribed(output_path):
    """Return True if an output file already exists and is non-empty."""
    return output_path.exists() and output_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def check_ffmpeg():
    """Exit with a helpful message if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: ffmpeg is not installed or not on PATH.\n"
            "       On macOS: brew install ffmpeg\n"
            "       On Linux:  sudo apt install ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)


def load_whisper_model(model_size):
    """Load and return the Whisper model, with a clear error on failure."""
    print(f"Loading Whisper model '{model_size}' (downloading on first use)...")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        print(f"ERROR: Failed to load Whisper model '{model_size}': {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Model loaded.\n")
    return model


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_file(model, audio_path):
    """Run Whisper inference and return the result dict."""
    return model.transcribe(str(audio_path))


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def seconds_to_srt_timestamp(seconds):
    """Convert float seconds to SRT timestamp string 'HH:MM:SS,mmm'."""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_as_txt(result):
    return result["text"].strip()


def format_as_json(result, audio_path):
    data = {
        "source_file": str(audio_path),
        "language": result.get("language", "unknown"),
        "text": result["text"].strip(),
        "segments": [
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def format_as_srt(result):
    lines = []
    for seg in result.get("segments", []):
        lines.append(str(seg["id"] + 1))
        start = seconds_to_srt_timestamp(seg["start"])
        end = seconds_to_srt_timestamp(seg["end"])
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def write_output(content, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def print_progress(current, total, audio_path, status):
    print(f"[{current}/{total}] {status}: {audio_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    check_ffmpeg()

    audio_files = resolve_input_path(args)
    total = len(audio_files)
    print(f"Found {total} audio file(s) to process.\n")

    model = load_whisper_model(args.model)

    transcribed = 0
    skipped = 0
    failed = 0

    for i, audio_path in enumerate(audio_files, start=1):
        output_path = get_output_path(audio_path, args)

        if already_transcribed(output_path):
            print_progress(i, total, audio_path, "Skipping (already transcribed)")
            skipped += 1
            continue

        print_progress(i, total, audio_path, "Transcribing")

        try:
            result = transcribe_file(model, audio_path)
        except PermissionError:
            print(f"  [WARN] Permission denied: {audio_path.name} - skipping.", file=sys.stderr)
            failed += 1
            continue
        except Exception as e:
            print(f"  [ERROR] Failed to transcribe {audio_path.name}: {e}", file=sys.stderr)
            failed += 1
            continue

        if args.format == "txt":
            content = format_as_txt(result)
        elif args.format == "json":
            content = format_as_json(result, audio_path)
        else:
            content = format_as_srt(result)

        try:
            write_output(content, output_path)
        except OSError as e:
            print(f"  [ERROR] Could not write {output_path}: {e}", file=sys.stderr)
            failed += 1
            continue

        print(f"        -> {output_path}")
        transcribed += 1

    print(f"\nDone. {transcribed} transcribed, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
