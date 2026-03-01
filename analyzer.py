"""
analyzer.py

Analyze transcribed text to extract key points and organize into categories.
Uses the OpenAI ChatCompletion API (GPT) for intelligent analysis.
Falls back to a basic extractive approach when no API key is available.
"""

import json
import os
import re
from dataclasses import dataclass, field

# Optional dependency — gracefully degrade when unavailable.
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class AnalysisResult:
    """Structured output from transcript analysis."""

    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    categories: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "key_points": self.key_points,
            "categories": self.categories,
        }

    def to_markdown(self) -> str:
        parts: list[str] = []

        if self.summary:
            parts.append("## Summary\n")
            parts.append(self.summary + "\n")

        if self.key_points:
            parts.append("\n## Key Points\n")
            for point in self.key_points:
                parts.append(f"- {point}")
            parts.append("")

        if self.categories:
            parts.append("\n## Categorized Notes\n")
            for category, items in self.categories.items():
                parts.append(f"### {category}\n")
                for item in items:
                    parts.append(f"- {item}")
                parts.append("")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenAI-powered analysis
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert note-taking assistant. You receive a transcript of a voice \
memo and produce structured notes.

Respond ONLY with valid JSON matching this schema (no markdown fences):
{
  "summary": "A concise 2-3 sentence summary of the entire memo.",
  "key_points": [
    "First key takeaway or highlight …",
    "Second key takeaway …"
  ],
  "categories": {
    "Category Name": [
      "Bullet point belonging to this category …"
    ]
  }
}

Guidelines:
- Identify 3-10 key points depending on memo length.
- Choose 2-6 broad, descriptive category names that fit the content \
(e.g. "Action Items", "Decisions", "Ideas", "Questions", "Follow-ups", \
"People Mentioned", "Dates & Deadlines", "Technical Details", etc.).
- Each category should contain 1-5 concise bullets.
- Be faithful to the original content — do not invent information.
"""


def _analyze_with_openai(transcript: str, api_key: str | None = None) -> AnalysisResult:
    """Call the OpenAI API to extract key points and categories."""
    if OpenAI is None:
        raise ImportError("The 'openai' package is required. Install with: pip install openai")

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the voice memo transcript:\n\n{transcript}"},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)

    return AnalysisResult(
        summary=data.get("summary", ""),
        key_points=data.get("key_points", []),
        categories=data.get("categories", {}),
    )


# ---------------------------------------------------------------------------
# Basic fallback analysis (no API key required)
# ---------------------------------------------------------------------------

_FILLER_WORDS = {
    "um", "uh", "like", "you know", "i mean", "so", "well", "actually",
    "basically", "right", "okay", "ok",
}


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


def _score_sentence(sentence: str) -> float:
    """Heuristic score: longer, non-filler sentences score higher."""
    words = sentence.lower().split()
    if len(words) < 4:
        return 0.0
    filler_count = sum(1 for w in words if w in _FILLER_WORDS)
    return len(words) - filler_count * 2


def _analyze_basic(transcript: str) -> AnalysisResult:
    """Extract key points using simple heuristics (no LLM needed)."""
    sentences = _split_sentences(transcript)
    if not sentences:
        return AnalysisResult(summary="(empty transcript)")

    scored = sorted(
        [(s, _score_sentence(s)) for s in sentences],
        key=lambda x: x[1],
        reverse=True,
    )

    # Summary: first 2 sentences
    summary = " ".join(sentences[:2])

    # Key points: top-scoring unique sentences
    seen = set()
    key_points: list[str] = []
    for sent, score in scored:
        if score <= 0:
            continue
        norm = sent.lower()
        if norm not in seen:
            seen.add(norm)
            key_points.append(sent)
        if len(key_points) >= 7:
            break

    # Basic categorization by keyword detection
    categories: dict[str, list[str]] = {}
    action_kw = {"need to", "should", "must", "will", "going to", "have to", "plan to", "todo", "to do"}
    question_kw = {"?"}
    idea_kw = {"idea", "maybe", "what if", "could", "might", "suggest", "proposal", "consider"}
    date_kw = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
               "january", "february", "march", "april", "may", "june", "july", "august",
               "september", "october", "november", "december", "deadline", "due", "by"}

    for sent in sentences:
        lower = sent.lower()
        if any(kw in lower for kw in action_kw):
            categories.setdefault("Action Items", []).append(sent)
        if any(kw in sent for kw in question_kw):
            categories.setdefault("Questions", []).append(sent)
        if any(kw in lower for kw in idea_kw):
            categories.setdefault("Ideas & Suggestions", []).append(sent)
        if any(kw in lower for kw in date_kw):
            categories.setdefault("Dates & Deadlines", []).append(sent)

    # Limit each category to 5 items
    categories = {k: v[:5] for k, v in categories.items()}

    if not categories:
        categories["General Notes"] = key_points[:5] if key_points else sentences[:3]

    return AnalysisResult(
        summary=summary,
        key_points=key_points,
        categories=categories,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(transcript: str, api_key: str | None = None, use_openai: bool = True) -> AnalysisResult:
    """
    Analyze a transcript to extract key points and categorized notes.

    Args:
        transcript: The full text transcript to analyze.
        api_key: Optional OpenAI API key. Falls back to OPENAI_API_KEY env var.
        use_openai: If True (default), attempt OpenAI analysis first.
                    Falls back to basic analysis on failure.

    Returns:
        AnalysisResult with summary, key_points, and categories.
    """
    if not transcript or not transcript.strip():
        return AnalysisResult(summary="(empty transcript)")

    if use_openai:
        effective_key = api_key or os.environ.get("OPENAI_API_KEY")
        if effective_key and OpenAI is not None:
            try:
                return _analyze_with_openai(transcript, api_key=effective_key)
            except Exception as e:
                print(f"[WARN] OpenAI analysis failed ({e}), falling back to basic analysis.")

    return _analyze_basic(transcript)
