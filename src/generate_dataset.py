import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_data(rows, noise, seed):
    rng = np.random.default_rng(seed)

    study_hours = rng.uniform(0.5, 10, rows)
    attendance = rng.uniform(50, 100, rows)
    previous_gpa = rng.uniform(1.5, 4.0, rows)
    assignments = rng.uniform(40, 100, rows)
    sleep_hours = rng.uniform(4, 10, rows)

    # Standardize features around approximate midpoints.
    score = (
        0.8 * (study_hours - 5)
        + 0.045 * (attendance - 75)
        + 1.2 * (previous_gpa - 2.7)
        + 0.035 * (assignments - 70)
        + 0.15 * (sleep_hours - 7)
    )

    # Convert the score into a probability between 0 and 1.
    probability = 1 / (1 + np.exp(-score / 2.5))

    success = rng.binomial(1, probability)

    # Flip a proportion of labels to simulate noisy outcomes.
    flip = rng.random(rows) < noise
    success[flip] = 1 - success[flip]

    return pd.DataFrame({
        "study_hours": study_hours.round(1),
        "attendance": attendance.round(1),
        "previous_gpa": previous_gpa.round(2),
        "assignments_completed": assignments.round(1),
        "sleep_hours": sleep_hours.round(1),
        "success": success
    })


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic student data."
    )
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rows < 50:
        parser.error("--rows must be at least 50.")

    if not 0 <= args.noise <= 0.5:
        parser.error("--noise must be between 0 and 0.5.")

    data = generate_data(args.rows, args.noise, args.seed)

    output = Path(__file__).resolve().parent.parent / "data" / "student_data.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)

    print(f"Generated {len(data)} synthetic records.")
    print(f"Label noise: {args.noise:.0%}")
    print("\nOutcome distribution:")
    print(data["success"].value_counts(normalize=True).sort_index().rename(
        index={0: "Not successful", 1: "Successful"}
    ).round(3))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
