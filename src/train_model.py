import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def main():
    print("Loading dataset...")

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

    print("\nFeatures:")
    print(features)

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("Training examples:", len(X_train))
    print("Testing examples:", len(X_test))

    # Build a machine-learning pipeline.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Model training complete.")

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel accuracy:")
    print(f"{accuracy:.2%}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    # Save trained model
    joblib.dump(model, "student_success_model.pkl")

    print("\nSaved model as:")
    print("student_success_model.pkl")


if __name__ == "__main__":
    main()
