"""
Tests for src/classifier/features.py

Covers the feature extraction logic that turns raw prompt text into
numeric features for the classifier. These are pure functions with no
external dependencies (no API calls, no model loading), so they run
instantly and should never be skipped.
"""
import pytest
from src.classifier.features import (
    extract_features,
    featurize,
    count_keywords,
    count_constraints,
    has_long_context,
    output_format_complexity,
    FEATURE_NAMES,
    COMPLEX_KEYWORDS,
    SIMPLE_KEYWORDS,
)


class TestCountKeywords:
    def test_finds_present_keyword(self):
        assert count_keywords("please design a system", COMPLEX_KEYWORDS) == 1

    def test_returns_zero_when_absent(self):
        assert count_keywords("what is the capital of france", COMPLEX_KEYWORDS) == 0

    def test_is_case_insensitive(self):
        assert count_keywords("DESIGN a system", COMPLEX_KEYWORDS) == 1

    def test_counts_multiple_matches(self):
        text = "design and propose a strategy"
        assert count_keywords(text, COMPLEX_KEYWORDS) >= 2


class TestCountConstraints:
    def test_no_constraints_in_simple_prompt(self):
        assert count_constraints("what is the capital of france") == 0

    def test_detects_constraint_language(self):
        assert count_constraints("design a system considering budget constraints") > 0

    def test_detects_numbered_list_markers(self):
        text = "1. first step 2. second step 3. third step"
        assert count_constraints(text) >= 3


class TestHasLongContext:
    def test_short_prompt_is_false(self):
        assert has_long_context("what is 2 plus 2") is False

    def test_long_prompt_is_true(self):
        assert has_long_context("x" * 500) is True

    def test_boundary_just_under_threshold(self):
        assert has_long_context("x" * 399) is False


class TestOutputFormatComplexity:
    def test_no_format_signal_returns_zero(self):
        assert output_format_complexity("what is the capital of france") == 0

    def test_simple_format_returns_one(self):
        assert output_format_complexity("give me a list of colors") == 1

    def test_complex_format_returns_two(self):
        assert output_format_complexity("design a database schema") == 2


class TestExtractFeatures:
    def test_returns_all_expected_keys(self):
        features = extract_features("What is the capital of France?")
        assert set(features.keys()) == set(FEATURE_NAMES)

    def test_simple_prompt_has_low_complex_keyword_count(self):
        features = extract_features("What is the capital of France?")
        assert features["complex_keyword_count"] == 0

    def test_complex_prompt_has_complex_keyword_signal(self):
        features = extract_features(
            "Design a distributed caching strategy considering latency tradeoffs"
        )
        assert features["complex_keyword_count"] > 0
        assert features["constraint_count"] > 0

    def test_word_count_is_accurate(self):
        features = extract_features("one two three four five")
        assert features["word_count"] == 5

    def test_empty_string_does_not_crash(self):
        features = extract_features("")
        assert features["word_count"] == 0
        assert features["char_count"] == 0

    def test_question_mark_detected(self):
        assert extract_features("Is this a question?")["has_question_mark"] == 1
        assert extract_features("This is a statement.")["has_question_mark"] == 0


class TestFeaturize:
    def test_returns_correct_number_of_rows(self):
        prompts = ["prompt one", "prompt two", "prompt three"]
        result = featurize(prompts)
        assert len(result) == 3

    def test_each_row_matches_feature_name_count(self):
        result = featurize(["a test prompt"])
        assert len(result[0]) == len(FEATURE_NAMES)

    def test_empty_list_returns_empty(self):
        assert featurize([]) == []
