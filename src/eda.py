"""
Exploratory Data Analysis for Student Performance Analysis.
Prints statistical findings and insights to the console.
"""
import pandas as pd
import numpy as np


def score_overview(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("SCORE OVERVIEW")
    print("=" * 60)
    score_cols = ["math_score", "reading_score", "writing_score", "average_score"]
    stats = df[score_cols].agg(["mean", "median", "std", "min", "max"]).round(2)
    print(stats.to_string())
    print()


def grade_distribution(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("GRADE DISTRIBUTION")
    print("=" * 60)
    counts = df["grade"].value_counts().sort_index()
    pct    = (counts / len(df) * 100).round(1)
    result = pd.DataFrame({"Count": counts, "Percentage (%)": pct})
    print(result.to_string())
    print()


def gender_analysis(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("GENDER ANALYSIS")
    print("=" * 60)
    score_cols = ["math_score", "reading_score", "writing_score", "average_score"]
    grp = df.groupby("gender")[score_cols].mean().round(2)
    print(grp.to_string())
    print()

    for col in score_cols:
        male_mean   = df[df["gender"] == "Male"][col].mean()
        female_mean = df[df["gender"] == "Female"][col].mean()
        diff = female_mean - male_mean
        higher = "Female" if diff > 0 else "Male"
        print(f"  {col}: {higher} score higher by {abs(diff):.2f} pts")
    print()


def study_hours_analysis(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("STUDY HOURS vs AVERAGE SCORE")
    print("=" * 60)
    if "study_category" in df.columns:
        grp = df.groupby("study_category", observed=True)["average_score"].agg(["mean", "count"]).round(2)
        grp.columns = ["Avg Score", "Students"]
        print(grp.to_string())
    corr = df["study_hours_per_day"].corr(df["average_score"])
    print(f"\n  Pearson correlation (study hours vs avg score): {corr:.4f}")
    print()


def attendance_analysis(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("ATTENDANCE RATE vs AVERAGE SCORE")
    print("=" * 60)
    if "attendance_category" in df.columns:
        grp = df.groupby("attendance_category", observed=True)["average_score"].agg(["mean", "count"]).round(2)
        grp.columns = ["Avg Score", "Students"]
        print(grp.to_string())
    corr = df["attendance_rate"].corr(df["average_score"])
    print(f"\n  Pearson correlation (attendance vs avg score): {corr:.4f}")
    print()


def sleep_analysis(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("SLEEP HOURS vs AVERAGE SCORE")
    print("=" * 60)
    if "sleep_category" in df.columns:
        grp = df.groupby("sleep_category", observed=True)["average_score"].agg(["mean", "count"]).round(2)
        grp.columns = ["Avg Score", "Students"]
        print(grp.to_string())
    corr = df["sleep_hours"].corr(df["average_score"])
    print(f"\n  Pearson correlation (sleep hours vs avg score): {corr:.4f}")
    print()


def parental_education_analysis(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("PARENTAL EDUCATION vs AVERAGE SCORE")
    print("=" * 60)
    grp = (
        df.groupby("parental_education", observed=True)["average_score"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    grp.columns = ["Mean Score", "Median Score", "Students"]
    print(grp.to_string())
    print()


def extracurricular_analysis(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("EXTRACURRICULAR ACTIVITIES vs AVERAGE SCORE")
    print("=" * 60)
    grp = df.groupby("extracurricular")["average_score"].agg(["mean", "std", "count"]).round(2)
    grp.index = grp.index.map({0: "No", 1: "Yes"})
    grp.columns = ["Mean Score", "Std Dev", "Students"]
    print(grp.to_string())
    print()


def internet_analysis(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("INTERNET ACCESS vs AVERAGE SCORE")
    print("=" * 60)
    grp = df.groupby("internet_access")["average_score"].agg(["mean", "std", "count"]).round(2)
    grp.index = grp.index.map({0: "No", 1: "Yes"})
    grp.columns = ["Mean Score", "Std Dev", "Students"]
    print(grp.to_string())
    print()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("CORRELATION MATRIX (Numeric Features)")
    print("=" * 60)
    num_cols = [
        "study_hours_per_day", "attendance_rate", "sleep_hours",
        "internet_access", "extracurricular",
        "math_score", "reading_score", "writing_score", "average_score",
    ]
    corr = df[num_cols].corr().round(3)
    print(corr.to_string())
    print()
    return corr


def top_bottom_performers(df: pd.DataFrame, n: int = 5) -> None:
    print("=" * 60)
    print(f"TOP {n} PERFORMERS")
    print("=" * 60)
    top = df.nlargest(n, "average_score")[
        ["student_id", "gender", "average_score", "grade", "study_hours_per_day", "attendance_rate"]
    ]
    print(top.to_string(index=False))
    print()

    print(f"BOTTOM {n} PERFORMERS")
    print("=" * 60)
    bottom = df.nsmallest(n, "average_score")[
        ["student_id", "gender", "average_score", "grade", "study_hours_per_day", "attendance_rate"]
    ]
    print(bottom.to_string(index=False))
    print()


def print_insights(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    overall_avg = df["average_score"].mean()
    pass_rate   = (df["average_score"] >= 55).mean() * 100
    a_rate      = (df["grade"] == "A").mean() * 100

    top_study   = df.groupby("study_category", observed=True)["average_score"].mean().idxmax()
    top_attend  = df.groupby("attendance_category", observed=True)["average_score"].mean().idxmax()

    study_corr  = df["study_hours_per_day"].corr(df["average_score"])
    attend_corr = df["attendance_rate"].corr(df["average_score"])

    internet_yes = df[df["internet_access"] == 1]["average_score"].mean()
    internet_no  = df[df["internet_access"] == 0]["average_score"].mean()

    extra_yes = df[df["extracurricular"] == 1]["average_score"].mean()
    extra_no  = df[df["extracurricular"] == 0]["average_score"].mean()

    print(f"  1. Overall average score      : {overall_avg:.1f}/100")
    print(f"  2. Pass rate (score >= 55)     : {pass_rate:.1f}%")
    print(f"  3. Students earning 'A' grade  : {a_rate:.1f}%")
    print(f"  4. Strongest study group       : {top_study}")
    print(f"  5. Strongest attendance group  : {top_attend}")
    print(f"  6. Study hours correlation     : {study_corr:.3f}  (higher study = higher score)")
    print(f"  7. Attendance correlation      : {attend_corr:.3f}  (higher attendance = higher score)")
    print(f"  8. Internet access impact      : +{internet_yes - internet_no:.1f} pts avg advantage")
    print(f"  9. Extracurricular impact      : +{extra_yes - extra_no:.1f} pts avg advantage")
    print()
