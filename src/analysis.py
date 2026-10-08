import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    data = pd.read_csv("data/student_data.csv")

    print("\nFirst five rows:")
    print(data.head())

    print("\nDataset shape:")
    print(data.shape)

    print("\nDataset information:")
    data.info()

    print("\nSummary statistics:")
    print(data.describe())

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nSuccess distribution:")
    print(data["success"].value_counts())

    print("\nAverage study hours by success:")
    print(data.groupby("success")["study_hours"].mean())

    print("\nAverage GPA by success:")
    print(data.groupby("success")["previous_gpa"].mean())

    print("\nCorrelation matrix:")
    print(data.corr(numeric_only=True))

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="success", y="study_hours", data=data)
    plt.title("Study Hours vs Academic Success")
    plt.xlabel("Success (0 = No, 1 = Yes)")
    plt.ylabel("Study Hours")
    plt.tight_layout()
    plt.savefig("study_hours_vs_success.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="success", y="previous_gpa", data=data)
    plt.title("Previous GPA vs Academic Success")
    plt.xlabel("Success (0 = No, 1 = Yes)")
    plt.ylabel("Previous GPA")
    plt.tight_layout()
    plt.savefig("gpa_vs_success.png")
    plt.show()

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        data.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.show()


if __name__ == "__main__":
    main()
