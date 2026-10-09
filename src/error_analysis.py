from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nConfusion Matrix")
    print("Rows = actual; columns = predicted")
    print("[[true 0, false positive],")
    print(" [false negative, true 1]]")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification Report")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not successful", "Successful"],
        zero_division=0
    ))

    errors = X_test.copy()
    errors["actual"] = y_test
    errors["predicted"] = predictions
    errors["error_type"] = "Correct"

    errors.loc[
        (errors["actual"] == 0) & (errors["predicted"] == 1),
        "error_type"
    ] = "False Positive"

    errors.loc[
        (errors["actual"] == 1) & (errors["predicted"] == 0),
        "error_type"
    ] = "False Negative"

    errors = errors[errors["actual"] != errors["predicted"]]
    output = ROOT / "prediction_errors.csv"
    errors.to_csv(output, index=False)

    print(f"\nIncorrect predictions: {len(errors)}")
    print(f"Error details saved to: {output}")
    print(errors.to_string(index=False))


if __name__ == "__main__":
    main()
