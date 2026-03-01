# CLAUDE.md

## Project Overview

A command-line tool to transcribe Apple Voice Memos (and any `.m4a`/`.caf` audio files) using OpenAI Whisper. Outputs transcripts in plain text, JSON (with timestamps), or SRT subtitle format.

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependency (macOS)
brew install ffmpeg

# Install system dependency (Linux)
sudo apt install ffmpeg
```

## Usage

```bash
# Auto-detect macOS Voice Memos folder
python voice_memo_transcriber.py

# Transcribe a single file
python voice_memo_transcriber.py --input memo.m4a

# Transcribe a directory, use a more accurate model, output JSON
python voice_memo_transcriber.py --input ~/recordings --model small --format json

# Output SRT subtitles to a dedicated folder
python voice_memo_transcriber.py --input memo.m4a --format srt --output ~/transcripts
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--input PATH` | auto-detect | Single `.m4a`/`.caf` file or directory of audio files |
| `--model SIZE` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `--output DIR` | same dir as audio | Directory to write transcript files |
| `--format` | `txt` | Output format: `txt`, `json`, or `srt` |

## Architecture

Single-file script (`voice_memo_transcriber.py`) organized into logical sections:

- **CLI** — argument parsing via `argparse`
- **Path resolution** — locates Voice Memos directory on macOS; resolves `--input` paths
- **Preflight checks** — validates `ffmpeg` is on PATH, loads Whisper model
- **Transcription** — runs Whisper inference on each audio file
- **Output formatting** — converts Whisper results to `txt`, `json`, or `srt`
- **Entry point** — orchestrates the pipeline, tracks transcribed/skipped/failed counts

## Key Details

- **Supported audio formats:** `.m4a`, `.caf`
- **Skips already-transcribed files** — output file must exist and be non-empty
- **macOS Voice Memos auto-detection** tries four candidate paths in order (newest to oldest OS version)
- **Whisper models** are downloaded on first use and cached locally by the `openai-whisper` library

## Dependencies

- `openai-whisper` — OpenAI's Whisper ASR model (see `requirements.txt`)
- `ffmpeg` — system package required by Whisper for audio decoding
