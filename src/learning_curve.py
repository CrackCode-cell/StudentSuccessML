from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    learning_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent


def main():
    data = pd.read_csv(ROOT / "data" / "student_data.csv")

    features = [
        "study_hours",
        "attendance",
        "previous_gpa",
        "assignments_completed",
        "sleep_hours"
    ]

    X = data[features]
    y = data["success"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    sizes, train_scores, validation_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        shuffle=True,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    results = pd.DataFrame({
        "Training examples": sizes,
        "Training accuracy": train_mean,
        "Validation accuracy": validation_mean
    })

    print("\nLearning Curve Results")
    print(results.round(3).to_string(index=False))

    results.to_csv(ROOT / "learning_curve_results.csv", index=False)

    plt.figure(figsize=(9, 6))
    plt.plot(sizes, train_mean, marker="o", label="Training accuracy")
    plt.plot(
        sizes,
        validation_mean,
        marker="o",
        label="Cross-validation accuracy"
    )

    plt.title("Learning Curve: Logistic Regression")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(ROOT / "learning_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
