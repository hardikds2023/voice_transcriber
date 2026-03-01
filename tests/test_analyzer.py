"""Tests for the analyzer module."""

import pytest

from analyzer import AnalysisResult, analyze, _analyze_basic, _split_sentences, _score_sentence


class TestSplitSentences:
    def test_simple(self):
        result = _split_sentences("Hello world. How are you? I'm fine.")
        assert result == ["Hello world.", "How are you?", "I'm fine."]

    def test_empty(self):
        assert _split_sentences("") == []

    def test_single_sentence(self):
        assert _split_sentences("Just one sentence.") == ["Just one sentence."]

    def test_no_punctuation(self):
        result = _split_sentences("No ending punctuation here")
        assert result == ["No ending punctuation here"]


class TestScoreSentence:
    def test_short_sentence_scores_zero(self):
        assert _score_sentence("Hi there.") == 0.0

    def test_filler_words_reduce_score(self):
        score_clean = _score_sentence("The project deadline is next Friday for the team.")
        score_filler = _score_sentence("Um like you know the project deadline is basically Friday.")
        assert score_clean > score_filler

    def test_long_sentence_scores_higher(self):
        short = _score_sentence("The meeting is tomorrow afternoon.")
        long = _score_sentence(
            "The meeting is tomorrow afternoon and we need to discuss the quarterly "
            "budget report and the new marketing strategy for the upcoming launch."
        )
        assert long > short


class TestAnalysisResult:
    def test_to_dict(self):
        result = AnalysisResult(
            summary="Test summary",
            key_points=["Point 1", "Point 2"],
            categories={"Action Items": ["Do this"]},
        )
        d = result.to_dict()
        assert d["summary"] == "Test summary"
        assert len(d["key_points"]) == 2
        assert "Action Items" in d["categories"]

    def test_to_markdown(self):
        result = AnalysisResult(
            summary="Test summary",
            key_points=["Point 1"],
            categories={"Ideas": ["Great idea"]},
        )
        md = result.to_markdown()
        assert "## Summary" in md
        assert "Test summary" in md
        assert "## Key Points" in md
        assert "- Point 1" in md
        assert "### Ideas" in md
        assert "- Great idea" in md

    def test_empty_result_markdown(self):
        result = AnalysisResult()
        md = result.to_markdown()
        assert md == ""


class TestAnalyzeBasic:
    def test_empty_transcript(self):
        result = _analyze_basic("")
        assert result.summary == "(empty transcript)"

    def test_extracts_action_items(self):
        text = (
            "We need to finish the report by Friday. "
            "The data looks good overall. "
            "I should call the client tomorrow."
        )
        result = _analyze_basic(text)
        assert "Action Items" in result.categories
        assert len(result.categories["Action Items"]) >= 1

    def test_extracts_questions(self):
        text = (
            "What should we do about the budget issue? "
            "The team seems concerned. "
            "Can we hire more people?"
        )
        result = _analyze_basic(text)
        assert "Questions" in result.categories
        assert len(result.categories["Questions"]) >= 1

    def test_extracts_ideas(self):
        text = (
            "Maybe we could try a different approach. "
            "What if we partner with another company. "
            "The current system works fine."
        )
        result = _analyze_basic(text)
        assert "Ideas & Suggestions" in result.categories

    def test_summary_is_populated(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        result = _analyze_basic(text)
        assert len(result.summary) > 0

    def test_key_points_limited(self):
        # Generate many sentences
        sentences = [f"This is important sentence number {i} about the project." for i in range(20)]
        text = " ".join(sentences)
        result = _analyze_basic(text)
        assert len(result.key_points) <= 7


class TestAnalyzePublicAPI:
    def test_empty_input(self):
        result = analyze("", use_openai=False)
        assert result.summary == "(empty transcript)"

    def test_whitespace_input(self):
        result = analyze("   ", use_openai=False)
        assert result.summary == "(empty transcript)"

    def test_basic_mode(self):
        text = (
            "Today we discussed the project timeline. "
            "We need to deliver the first milestone by March. "
            "The design team will provide mockups next week. "
            "I think we should consider using a new framework."
        )
        result = analyze(text, use_openai=False)
        assert result.summary
        assert isinstance(result.key_points, list)
        assert isinstance(result.categories, dict)

    def test_fallback_when_no_api_key(self):
        """Without an API key, analyze() should fall back to basic analysis."""
        import os
        original = os.environ.pop("OPENAI_API_KEY", None)
        try:
            result = analyze("Test transcript for analysis.", api_key=None, use_openai=True)
            assert isinstance(result, AnalysisResult)
        finally:
            if original is not None:
                os.environ["OPENAI_API_KEY"] = original
