import joblib
import pandas as pd


def main():
    print("Student Success Predictor")
    print("-------------------------")

    model = joblib.load("student_success_model.pkl")

    study_hours = float(input("Study hours per day: "))
    attendance = float(input("Attendance percentage: "))
    previous_gpa = float(input("Previous GPA: "))
    assignments_completed = float(
        input("Assignment completion percentage: ")
    )
    sleep_hours = float(input("Average sleep hours: "))

    student = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_gpa": previous_gpa,
        "assignments_completed": assignments_completed,
        "sleep_hours": sleep_hours
    }])

    prediction = model.predict(student)[0]
    probabilities = model.predict_proba(student)[0]

    probability_of_success = probabilities[1]

    print("\nInput summary:")
    print(f"Study hours: {study_hours}")
    print(f"Attendance: {attendance}%")
    print(f"Previous GPA: {previous_gpa}")
    print(f"Assignment completion: {assignments_completed}%")
    print(f"Sleep hours: {sleep_hours}")

    print("\nPrediction:")

    if prediction == 1:
        print("Likely to meet the success threshold.")
    else:
        print("Less likely to meet the success threshold.")

    print(
        f"Estimated probability of success: "
        f"{probability_of_success:.2%}"
    )


if __name__ == "__main__":
    main()
