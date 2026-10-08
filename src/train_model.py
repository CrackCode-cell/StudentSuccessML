import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel accuracy:")
    print(f"{accuracy:.2%}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    classifier = model.named_steps["classifier"]

    print("\nFeature importance:")
    for feature, coefficient in zip(features, classifier.coef_[0]):
        print(f"{feature}: {coefficient:.4f}")

    joblib.dump(model, "student_success_model.pkl")

    print("\nModel saved as student_success_model.pkl")


if __name__ == "__main__":
    main()
