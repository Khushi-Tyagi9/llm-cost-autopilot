import re

# Genuinely complex signals: strategy, design, judgment, multi-step planning,
# creative long-form generation. Removed "analyze" and "compare" - those
# alone don't distinguish tier 2 (structured analysis) from tier 3.
COMPLEX_KEYWORDS = [
    "design", "propose", "evaluate", "critique", "construct", "diagnose",
    "mediate", "justify", "strategy", "roadmap", "architecture",
    "tradeoff", "trade-off", "nuanced", "curriculum", "screenplay",
    "persuasive essay", "short story", "novel", "poem", "dialogue between",
    "ethical", "framework", "multi-step", "phased", "second-order",
]

SIMPLE_KEYWORDS = [
    "extract", "convert", "list", "what is", "who is", "reformat",
    "capital of", "how many", "what year",
]

MODERATE_KEYWORDS = [
    "summarize", "classify", "categorize", "organize", "break down",
    "analyze", "compare", "review", "improve", "suggest", "explain",
]

# Words that signal creative generation - a defining Tier 3 trait per brief,
# but ambiguous alone ("write a list" is Tier 1). Weighted separately.
CREATIVE_KEYWORDS = ["write a story", "write a poem", "write a screenplay",
                      "write a dialogue", "write a persuasive", "write a scene",
                      "write a satirical", "write an opening", "write a nuanced",
                      "write resume points", "write ats-friendly"]
UNGROUNDED_FACT_KEYWORDS = [
    "statistics", "stats", "cutoff", "percentage", "how many",
    "ranking", "rank", "spot round", "admission", "seats",
    "average", "population of", "release date", "score of",
    "results of", "when did", "who is the", "how much does",
]


def is_ungrounded_factual_query(text: str) -> bool:
    """Distinguishes queries asking the model to recall specific facts
    from its own training data (higher fabrication risk) from queries
    that provide source material to work from, like a pasted article
    or document (lower fabrication risk, since the model is processing
    given content rather than recalling unaided).

    A query is treated as 'grounded' (safer) if it includes a
    substantial block of provided text to work from - long pasted
    content is the strongest signal the model has something real to
    reference rather than needing to recall facts unaided.
    """
    has_long_pasted_content = has_long_context(text)
    if has_long_pasted_content:
        return False  # grounded - has source material, lower risk

    return count_keywords(text, UNGROUNDED_FACT_KEYWORDS) >= 1


def count_keywords(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def count_constraints(text: str) -> int:
    signals = ["must", "should", "considering", "given", "based on",
               "while", "without", "balancing", "constraint", "minimizing",
               "maximizing", "given that"]
    text_lower = text.lower()
    count = sum(1 for s in signals if s in text_lower)
    count += len(re.findall(r"\d+\.", text))
    return count


def has_long_context(text: str) -> bool:
    return len(text) > 400


def output_format_complexity(text: str) -> int:
    text_lower = text.lower()
    if any(w in text_lower for w in ["schema", "screenplay", "architecture", "curriculum", "multi-step"]):
        return 2
    if any(w in text_lower for w in ["list", "bullet", "table", "steps"]):
        return 1
    return 0


def extract_features(prompt: str) -> dict:
    word_count = len(prompt.split())
    char_count = len(prompt)

    return {
        "word_count": word_count,
        "char_count": char_count,
        "complex_keyword_count": count_keywords(prompt, COMPLEX_KEYWORDS),
        "simple_keyword_count": count_keywords(prompt, SIMPLE_KEYWORDS),
        "moderate_keyword_count": count_keywords(prompt, MODERATE_KEYWORDS),
        "creative_keyword_count": count_keywords(prompt, CREATIVE_KEYWORDS),
        "constraint_count": count_constraints(prompt),
        "has_long_context": int(has_long_context(prompt)),
        "output_format_complexity": output_format_complexity(prompt),
        "has_question_mark": int("?" in prompt),
        "sentence_count": prompt.count(".") + prompt.count("?") + prompt.count("!"),
        "is_short_prompt": int(word_count <= 8),
    }


FEATURE_NAMES = [
    "word_count", "char_count", "complex_keyword_count",
    "simple_keyword_count", "moderate_keyword_count", "creative_keyword_count",
    "constraint_count", "has_long_context", "output_format_complexity",
    "has_question_mark", "sentence_count", "is_short_prompt",
]


def featurize(prompts: list[str]):
    rows = []
    for p in prompts:
        feats = extract_features(p)
        rows.append([feats[name] for name in FEATURE_NAMES])
    return rows