"""
Trains the complexity classifier on data/prompts_labeled.csv.
Uses 5-fold cross-validation for a reliable accuracy estimate,
then trains a final model on all data for saving.

Usage: python -m src.classifier.train
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from src.classifier.features import featurize, FEATURE_NAMES


def load_data(csv_path="data/prompts_labeled.csv"):
    df = pd.read_csv(csv_path)
    return df["prompt"].tolist(), df["tier"].tolist()


def main():
    prompts, tiers = load_data()
    print(f"Loaded {len(prompts)} labeled prompts.")

    X = np.array(featurize(prompts))
    y = np.array(tiers)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Cross-validated accuracy (reliable estimate) ---
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=cv)
    print(f"\nRandom Forest 5-fold CV: {rf_scores.mean():.2%} (scores: {[f'{s:.2%}' for s in rf_scores]})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_scaled, y, cv=cv)
    print(f"Logistic Regression 5-fold CV: {lr_scores.mean():.2%} (scores: {[f'{s:.2%}' for s in lr_scores]})")

    # --- Held-out test set, for a final human-readable report ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    best_is_rf = rf_scores.mean() >= lr_scores.mean()

    if best_is_rf:
        print(f"\nRandom Forest selected (CV mean {rf_scores.mean():.2%} vs {lr_scores.mean():.2%})")
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        print(classification_report(y_test, preds))
        final_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
        final_model.fit(X, y)  # retrain on ALL data for the saved model
        needs_scaler = False
        scaler_to_save = None
    else:
        print(f"\nLogistic Regression selected (CV mean {lr_scores.mean():.2%} vs {rf_scores.mean():.2%})")
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        lr.fit(X_train_s, y_train)
        preds = lr.predict(X_test_s)
        print(classification_report(y_test, preds))
        final_scaler = StandardScaler()
        X_all_scaled = final_scaler.fit_transform(X)
        final_model = LogisticRegression(max_iter=1000, random_state=42)
        final_model.fit(X_all_scaled, y)
        needs_scaler = True
        scaler_to_save = final_scaler

    joblib.dump({
        "model": final_model,
        "scaler": scaler_to_save,
        "needs_scaler": needs_scaler,
        "feature_names": FEATURE_NAMES,
    }, "src/classifier/model.pkl")
    print("\nSaved final model (trained on all data) to src/classifier/model.pkl")


if __name__ == "__main__":
    main()