from src.classifier.features import contains_hedging_language


def test_detects_explicit_uncertainty():
    assert contains_hedging_language("I don't have information on this topic.") is True


def test_detects_verification_disclaimer():
    assert contains_hedging_language("Please note that this data may not be accurate.") is True


def test_confident_answer_not_flagged():
    assert contains_hedging_language("The capital of France is Paris.") is False


def test_case_insensitive():
    assert contains_hedging_language("I CANNOT VERIFY this information.") is True