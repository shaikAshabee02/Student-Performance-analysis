"""
Data loading and preprocessing utilities for Student Performance Analysis.
"""
import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def basic_info(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Duplicate rows : {df.duplicated().sum()}")
    print()

    print("--- Column Types ---")
    print(df.dtypes.to_string())
    print()

    print("--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(missing[missing > 0].to_string())
    print()


def descriptive_stats(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS (Numeric Columns)")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    print(df[numeric_cols].describe().round(2).to_string())
    print()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna()

    # Ensure score columns are within valid range
    for col in ["math_score", "reading_score", "writing_score", "average_score"]:
        if col in df.columns:
            df = df[(df[col] >= 0) & (df[col] <= 100)]

    # Recompute average score to stay consistent
    df["average_score"] = (
        df["math_score"] + df["reading_score"] + df["writing_score"]
    ) / 3
    df["average_score"] = df["average_score"].round(1)

    # Ordered category for parental education
    edu_order = [
        "No Formal Education",
        "Primary School",
        "High School",
        "Bachelor's Degree",
        "Master's Degree",
    ]
    df["parental_education"] = pd.Categorical(
        df["parental_education"], categories=edu_order, ordered=True
    )

    # Ordered category for grade
    grade_order = ["A", "B", "C", "D", "F"]
    df["grade"] = pd.Categorical(df["grade"], categories=grade_order, ordered=True)

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["study_category"] = pd.cut(
        df["study_hours_per_day"],
        bins=[0, 2, 4, 6, 8.1],
        labels=["Low (0-2h)", "Moderate (2-4h)", "High (4-6h)", "Very High (6-8h)"],
    )
    df["attendance_category"] = pd.cut(
        df["attendance_rate"],
        bins=[0, 65, 75, 85, 100.1],
        labels=["Poor (<65%)", "Average (65-75%)", "Good (75-85%)", "Excellent (>85%)"],
    )
    df["sleep_category"] = pd.cut(
        df["sleep_hours"],
        bins=[0, 5.5, 7, 8.5, 11],
        labels=["Insufficient (<5.5h)", "Adequate (5.5-7h)", "Optimal (7-8.5h)", "Excess (>8.5h)"],
    )
    return df
