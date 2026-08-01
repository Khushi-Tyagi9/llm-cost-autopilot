from src.classifier.features import is_ungrounded_factual_query


def test_ungrounded_factual_query_detected():
    assert is_ungrounded_factual_query("Spot round statistics for FC/AD/FD in top 7 NIFTs") is True


def test_grounded_query_with_long_content_not_flagged():
    long_text = "Summarize this: " + ("x " * 300)
    assert is_ungrounded_factual_query(long_text) is False


def test_simple_query_without_risk_keywords_not_flagged():
    assert is_ungrounded_factual_query("Write a poem about the ocean") is False