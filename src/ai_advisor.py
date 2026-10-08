import joblib
import pandas as pd


def main():
    model = joblib.load("student_success_model.pkl")

    print("Student AI Learning Advisor")
    print("---------------------------")

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
    probability = model.predict_proba(student)[0][1]

    print("\nAI-style analysis")
    print("-----------------")

    if prediction == 1:
        print("The model predicts a positive academic outcome.")
    else:
        print("The model predicts a lower likelihood of a positive outcome.")

    print(f"Model confidence: {probability:.2%}")

    if attendance < 80:
        print("Suggestion: Improving attendance may be helpful.")

    if assignments_completed < 75:
        print("Suggestion: Completing more assignments may improve performance.")

    if study_hours < 4:
        print("Suggestion: Increasing consistent study time may help.")

    if sleep_hours < 6:
        print("Suggestion: Getting more consistent sleep may support academic performance.")

    if previous_gpa >= 3.5:
        print("Strength: Previous GPA is relatively strong.")

    print("\nNote:")
    print("This is an educational AI/ML demonstration.")
    print("The model does not determine actual academic outcomes.")


if __name__ == "__main__":
    main()
