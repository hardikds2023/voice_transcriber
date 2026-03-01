"""Tests for the transcriber module (unit tests that don't require whisper)."""

import pytest
import sys
from unittest.mock import MagicMock

# Mock whisper before importing transcriber so tests work without GPU/whisper installed
sys.modules.setdefault("whisper", MagicMock())

from transcriber import (
    SUPPORTED_EXTENSIONS,
    format_timestamp,
)


class TestSupportedExtensions:
    def test_m4a_supported(self):
        assert ".m4a" in SUPPORTED_EXTENSIONS

    def test_caf_supported(self):
        assert ".caf" in SUPPORTED_EXTENSIONS

    def test_wav_supported(self):
        assert ".wav" in SUPPORTED_EXTENSIONS

    def test_mp3_supported(self):
        assert ".mp3" in SUPPORTED_EXTENSIONS


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00"

    def test_seconds_only(self):
        assert format_timestamp(45) == "00:45"

    def test_minutes_and_seconds(self):
        assert format_timestamp(125) == "02:05"

    def test_hours(self):
        assert format_timestamp(3661) == "01:01:01"

    def test_rounding(self):
        assert format_timestamp(59.6) == "01:00"

    def test_exact_minute(self):
        assert format_timestamp(60) == "01:00"

    def test_exact_hour(self):
        assert format_timestamp(3600) == "01:00:00"
