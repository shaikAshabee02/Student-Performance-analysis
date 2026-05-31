"""
Student Performance Analysis
=============================
Full pipeline: load → clean → EDA → visualize → insights.

Usage:
    python analysis.py                   # uses default data/student_data.csv
    python analysis.py --data <path>     # custom CSV path
    python analysis.py --no-plots        # skip chart generation
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import load_data, basic_info, descriptive_stats, clean_data, engineer_features
from src.eda import (
    score_overview, grade_distribution, gender_analysis,
    study_hours_analysis, attendance_analysis, sleep_analysis,
    parental_education_analysis, extracurricular_analysis,
    internet_analysis, correlation_matrix, top_bottom_performers,
    print_insights,
)
from src.visualizations import (
    plot_score_distributions, plot_grade_distribution,
    plot_gender_analysis, plot_study_hours, plot_attendance,
    plot_sleep_hours, plot_parental_education, plot_extra_internet,
    plot_correlation_heatmap, plot_scores_by_grade, plot_dashboard,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Student Performance Analysis")
    parser.add_argument("--data",     default="data/student_data.csv", help="Path to CSV dataset")
    parser.add_argument("--no-plots", action="store_true",             help="Skip generating plots")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load & Validate ──────────────────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"[ERROR] Dataset not found: {args.data}")
        print("  Run  python generate_data.py  first to create the dataset.")
        sys.exit(1)

    print(f"\nLoading dataset from: {args.data}")
    df_raw = load_data(args.data)

    # ── Overview ─────────────────────────────────────────────────────────────
    basic_info(df_raw)
    descriptive_stats(df_raw)

    # ── Clean & Engineer ─────────────────────────────────────────────────────
    df = clean_data(df_raw)
    df = engineer_features(df)
    print(f"Cleaned dataset shape: {df.shape}\n")

    # ── EDA ──────────────────────────────────────────────────────────────────
    score_overview(df)
    grade_distribution(df)
    gender_analysis(df)
    study_hours_analysis(df)
    attendance_analysis(df)
    sleep_analysis(df)
    parental_education_analysis(df)
    extracurricular_analysis(df)
    internet_analysis(df)
    correlation_matrix(df)
    top_bottom_performers(df, n=5)
    print_insights(df)

    # ── Visualizations ───────────────────────────────────────────────────────
    if not args.no_plots:
        print("Generating charts …")
        plot_score_distributions(df)
        plot_grade_distribution(df)
        plot_gender_analysis(df)
        plot_study_hours(df)
        plot_attendance(df)
        plot_sleep_hours(df)
        plot_parental_education(df)
        plot_extra_internet(df)
        plot_correlation_heatmap(df)
        plot_scores_by_grade(df)
        plot_dashboard(df)
        print(f"\nAll charts saved to the 'plots/' directory.\n")
    else:
        print("Plot generation skipped (--no-plots).\n")


if __name__ == "__main__":
    main()
