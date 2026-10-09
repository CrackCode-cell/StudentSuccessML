import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def main():
    # 1. Load the dataset
    data = pd.read_csv("data/student_data.csv")

    features = [
        "study_hours",
        "attendance",
        "previous_gpa",
        "assignments_completed",
        "sleep_hours"
    ]

    X = data[features]
    y = data["success"]

    # 2. Define three models
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )
    }

    # 3. Evaluate each model using five-fold cross-validation
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }

    results = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cross_validation,
            scoring=scoring
        )

        results.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1 Score": scores["test_f1"].mean()
        })

    # 4. Display and save the results
    results_df = pd.DataFrame(results)

    print("\nModel Comparison")
    print("================")
    print(results_df.round(3).to_string(index=False))

    results_df.to_csv("model_comparison.csv", index=False)

    # 5. Visualize model performance
    chart_data = results_df.set_index("Model")

    chart_data.plot(
        kind="bar",
        figsize=(11, 6),
        ylim=(0, 1),
        rot=0
    )

    plt.title("Model Performance Comparison")
    plt.ylabel("Cross-validation score")
    plt.xlabel("Machine learning model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
