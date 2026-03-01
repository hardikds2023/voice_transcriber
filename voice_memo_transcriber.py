#!/usr/bin/env python3
"""
voice_memo_transcriber.py

Transcribe Apple Voice Memos (and any .m4a/.caf/.wav/.mp3 files) using OpenAI
Whisper, then extract key points and organize notes by category.

Usage:
    python voice_memo_transcriber.py [--input PATH] [--model SIZE]
                                     [--output DIR] [--format {txt,json,md}]
                                     [--no-analysis] [--openai-key KEY]

Examples:
    # Transcribe and analyze a single file
    python voice_memo_transcriber.py --input memo.m4a

    # Transcribe a directory, output markdown notes
    python voice_memo_transcriber.py --input ~/recordings --format md

    # Transcribe only (no key-point analysis)
    python voice_memo_transcriber.py --input memo.m4a --no-analysis

    # Use a specific OpenAI key for GPT-powered analysis
    python voice_memo_transcriber.py --input memo.m4a --openai-key sk-...

    # Auto-detect macOS Voice Memos folder
    python voice_memo_transcriber.py

Requirements:
    pip install -r requirements.txt
    brew install ffmpeg   # (macOS)  /  sudo apt install ffmpeg  (Linux)
"""

import argparse
import json
import sys
from pathlib import Path

from transcriber import (
    SUPPORTED_EXTENSIONS,
    format_timestamp,
    load_model,
    transcribe,
)
from analyzer import analyze


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
        description="Transcribe Apple Voice Memos and extract key points.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python voice_memo_transcriber.py --input memo.m4a
  python voice_memo_transcriber.py --input ~/recordings --model small
  python voice_memo_transcriber.py --input memo.m4a --format md
  python voice_memo_transcriber.py --output ~/transcripts --format json
        """,
    )
    parser.add_argument(
        "--input",
        metavar="PATH",
        help=(
            "Path to a single audio file or a directory of audio files. "
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
        help="Directory to write output files. Defaults to the audio file's directory.",
    )
    parser.add_argument(
        "--format",
        default="md",
        choices=["txt", "json", "md"],
        help="Output format: plain text, JSON, or Markdown with analysis (default: md).",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip key-point extraction and categorization.",
    )
    parser.add_argument(
        "--openai-key",
        metavar="KEY",
        help="OpenAI API key for GPT-powered analysis. Falls back to OPENAI_API_KEY env var.",
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
                    f"ERROR: Unsupported file type '{p.suffix}'. "
                    f"Expected: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                    file=sys.stderr,
                )
                sys.exit(1)
            return [p]
        files = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print(f"ERROR: No supported audio files found in {p}", file=sys.stderr)
            sys.exit(1)
        return files

    voice_memos_dir = detect_voice_memos_dir()
    if voice_memos_dir is None:
        print(
            "ERROR: Could not find the Voice Memos directory.\n"
            "       Pass --input to specify a path.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Auto-detected Voice Memos folder: {voice_memos_dir}")
    files = sorted(
        f for f in voice_memos_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"ERROR: No supported audio files found in {voice_memos_dir}", file=sys.stderr)
        sys.exit(1)
    return files


def get_output_path(audio_path, args):
    """Compute the output file path for a given audio file."""
    ext_map = {"txt": ".txt", "json": ".json", "md": ".md"}
    ext = ext_map[args.format]
    output_dir = Path(args.output).expanduser() if args.output else audio_path.parent
    return output_dir / (audio_path.stem + ext)


def already_processed(output_path):
    """Return True if an output file already exists and is non-empty."""
    return output_path.exists() and output_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_as_txt(result, analysis_result=None):
    """Plain text output with optional analysis."""
    parts = [result["text"]]

    if analysis_result:
        parts.append("\n" + "=" * 60)
        parts.append("SUMMARY")
        parts.append("=" * 60)
        parts.append(analysis_result.summary)

        parts.append("\n" + "-" * 60)
        parts.append("KEY POINTS")
        parts.append("-" * 60)
        for i, point in enumerate(analysis_result.key_points, 1):
            parts.append(f"  {i}. {point}")

        if analysis_result.categories:
            parts.append("\n" + "-" * 60)
            parts.append("CATEGORIZED NOTES")
            parts.append("-" * 60)
            for category, items in analysis_result.categories.items():
                parts.append(f"\n  [{category}]")
                for item in items:
                    parts.append(f"    - {item}")

    return "\n".join(parts)


def format_as_json(result, analysis_result=None):
    """JSON output with transcription and optional analysis."""
    data = {
        "source_file": result["source_file"],
        "language": result["language"],
        "text": result["text"],
        "segments": result["segments"],
    }
    if analysis_result:
        data["analysis"] = analysis_result.to_dict()
    return json.dumps(data, indent=2, ensure_ascii=False)


def format_as_md(result, analysis_result=None):
    """Markdown output with analysis and timestamped transcript."""
    source = Path(result["source_file"]).name
    parts = [f"# Voice Memo Notes: {source}\n"]

    if analysis_result:
        parts.append(analysis_result.to_markdown())

    parts.append("\n---\n")
    parts.append("## Full Transcript\n")

    if result["segments"]:
        for seg in result["segments"]:
            ts = format_timestamp(seg["start"])
            parts.append(f"**[{ts}]** {seg['text']}\n")
    else:
        parts.append(result["text"])

    return "\n".join(parts)


def write_output(content, output_path):
    """Write content to file, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    audio_files = resolve_input_path(args)
    total = len(audio_files)
    print(f"Found {total} audio file(s) to process.\n")

    print(f"Loading Whisper model '{args.model}'...")
    try:
        model = load_model(args.model)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print("Model loaded.\n")

    processed = 0
    skipped = 0
    failed = 0

    for i, audio_path in enumerate(audio_files, start=1):
        output_path = get_output_path(audio_path, args)

        if already_processed(output_path):
            print(f"[{i}/{total}] Skipping (already exists): {audio_path.name}")
            skipped += 1
            continue

        print(f"[{i}/{total}] Transcribing: {audio_path.name}")

        try:
            result = transcribe(model, audio_path)
        except PermissionError:
            print(f"  [WARN] Permission denied: {audio_path.name}", file=sys.stderr)
            failed += 1
            continue
        except Exception as e:
            print(f"  [ERROR] Transcription failed: {e}", file=sys.stderr)
            failed += 1
            continue

        # Analysis
        analysis_result = None
        if not args.no_analysis:
            print(f"         Analyzing transcript...")
            try:
                analysis_result = analyze(
                    result["text"],
                    api_key=args.openai_key,
                )
            except Exception as e:
                print(f"  [WARN] Analysis failed ({e}), saving transcript only.", file=sys.stderr)

        # Format output
        if args.format == "txt":
            content = format_as_txt(result, analysis_result)
        elif args.format == "json":
            content = format_as_json(result, analysis_result)
        else:
            content = format_as_md(result, analysis_result)

        try:
            write_output(content, output_path)
        except OSError as e:
            print(f"  [ERROR] Could not write {output_path}: {e}", file=sys.stderr)
            failed += 1
            continue

        print(f"         -> {output_path}")
        processed += 1

    print(f"\nDone. {processed} processed, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
