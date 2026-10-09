from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "student_data.csv"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_gpa",
    "assignments_completed",
    "sleep_hours",
]


def main():
    data = pd.read_csv(DATA_PATH).dropna(
        subset=FEATURES + ["success"]
    )

    X = data[FEATURES]
    y = data["success"]

    if y.nunique() != 2:
        raise ValueError("Dataset must contain both outcome classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        ),
    }

    comparison = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        comparison.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
        })

        print(f"\n{name}")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
        print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
        print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.3f}")
        print(f"F1 Score: {f1_score(y_test, predictions, zero_division=0):.3f}")

    results = pd.DataFrame(comparison)
    results.to_csv(ROOT / "model_comparison.csv", index=False)

    print("\nModel Comparison")
    print(results.round(3).to_string(index=False))

    forest = models["Random Forest"]

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": forest.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importance.to_csv(ROOT / "feature_importance.csv", index=False)
    joblib.dump(forest, ROOT / "random_forest_model.pkl")

    print("\nRandom Forest Feature Importance")
    print(importance.round(3).to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.barh(importance["Feature"], importance["Importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Relative feature importance")
    plt.title("Random Forest: Feature Importance")
    plt.tight_layout()
    plt.savefig(ROOT / "feature_importance.png", dpi=150)
    plt.close()

    print("\nSaved model comparison, feature importance, graph, and model.")


if __name__ == "__main__":
    main()
