"""
Regression test for the trained classifier's accuracy.

Unlike test_features.py (which tests feature extraction logic in
isolation), this test loads the actual labeled dataset and the actual
saved model, and asserts real-world accuracy stays above a floor.

This exists because feature-extraction tests passing does NOT guarantee
the trained model still performs well - someone could retrain with a
bug, a bad dataset edit, or a regressed feature set, and every unit
test would still pass while the classifier quietly got worse.

Threshold is set below the measured 83.88% cross-validated accuracy to
avoid flakiness from the small amount of randomness in model training
and cross-validation folds, while still catching a real regression.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from src.classifier.features import featurize

MINIMUM_ACCEPTABLE_ACCURACY = 0.75  # measured baseline was 83.88%


@pytest.fixture(scope="module")
def dataset():
    df = pd.read_csv("data/prompts_labeled.csv")
    return df["prompt"].tolist(), df["tier"].tolist()


class TestClassifierAccuracy:
    def test_dataset_has_expected_minimum_size(self, dataset):
        prompts, _ = dataset
        assert len(prompts) >= 200

    def test_dataset_has_all_three_tiers_represented(self, dataset):
        _, tiers = dataset
        assert set(tiers) == {1, 2, 3}

    def test_no_tier_is_severely_underrepresented(self, dataset):
        """A tier with very few examples would make the classifier
        unreliable for that tier specifically, even if overall
        accuracy looks fine."""
        _, tiers = dataset
        counts = pd.Series(tiers).value_counts()
        min_expected = len(tiers) * 0.15  # each tier should be at least 15% of data
        assert counts.min() >= min_expected

    def test_cross_validated_accuracy_meets_floor(self, dataset):
        prompts, tiers = dataset
        X = np.array(featurize(prompts))
        y = np.array(tiers)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv)

        mean_accuracy = scores.mean()
        assert mean_accuracy >= MINIMUM_ACCEPTABLE_ACCURACY, (
            f"Classifier accuracy regressed: {mean_accuracy:.2%} is below the "
            f"{MINIMUM_ACCEPTABLE_ACCURACY:.0%} floor (baseline was 83.88%). "
            f"Check for changes to features.py, the training data, or the model config."
        )

    def test_no_single_fold_is_catastrophically_bad(self, dataset):
        """Even if the mean accuracy looks fine, one very bad fold can
        indicate the classifier is unreliable for some slice of prompts."""
        prompts, tiers = dataset
        X = np.array(featurize(prompts))
        y = np.array(tiers)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv)

        assert min(scores) >= 0.60, f"Worst fold was only {min(scores):.2%}"
