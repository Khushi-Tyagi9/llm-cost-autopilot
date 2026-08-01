from src.classifier.features import is_factual_risk


def test_detects_statistics_query():
    assert is_factual_risk("Spot round statistics for FC/AD/FD in top 7 NIFTs") is True


def test_detects_percentage_query():
    assert is_factual_risk("What percentage of students got placed?") is True


def test_simple_factual_query_not_flagged():
    assert is_factual_risk("What is the capital of France?") is False


def test_creative_query_not_flagged():
    assert is_factual_risk("Write a poem about the ocean") is False