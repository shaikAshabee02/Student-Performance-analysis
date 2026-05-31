"""
All matplotlib visualizations for Student Performance Analysis.
Charts are saved to the plots/ directory.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter

PLOTS_DIR = "plots"
PALETTE   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def _save(fig: plt.Figure, filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 1. Score Distributions ───────────────────────────────────────────────────
def plot_score_distributions(df: pd.DataFrame) -> None:
    score_cols = ["math_score", "reading_score", "writing_score", "average_score"]
    labels     = ["Math Score", "Reading Score", "Writing Score", "Average Score"]
    colors     = PALETTE[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Score Distributions", fontsize=16, fontweight="bold", y=1.01)

    for ax, col, label, color in zip(axes.flat, score_cols, labels, colors):
        data = df[col].dropna()
        ax.hist(data, bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(),   color="black",  linestyle="--", linewidth=1.2, label=f"Mean: {data.mean():.1f}")
        ax.axvline(data.median(), color="crimson", linestyle=":",  linewidth=1.2, label=f"Median: {data.median():.1f}")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Score")
        ax.set_ylabel("Number of Students")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "01_score_distributions.png")


# ── 2. Grade Distribution ─────────────────────────────────────────────────────
def plot_grade_distribution(df: pd.DataFrame) -> None:
    grade_order = ["A", "B", "C", "D", "F"]
    counts = df["grade"].value_counts().reindex(grade_order).fillna(0)
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Grade Distribution", fontsize=15, fontweight="bold")

    # Bar chart
    bars = ax1.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, int(val),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Grade")
    ax1.set_ylabel("Number of Students")
    ax1.set_title("Count per Grade")

    # Pie chart
    ax2.pie(
        counts.values,
        labels=[f"{g}\n({int(v)})" for g, v in zip(counts.index, counts.values)],
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    ax2.set_title("Grade Share (%)")

    plt.tight_layout()
    _save(fig, "02_grade_distribution.png")


# ── 3. Gender Analysis ────────────────────────────────────────────────────────
def plot_gender_analysis(df: pd.DataFrame) -> None:
    score_cols = ["math_score", "reading_score", "writing_score", "average_score"]
    labels     = ["Math", "Reading", "Writing", "Average"]

    male_means   = [df[df["gender"] == "Male"][c].mean()   for c in score_cols]
    female_means = [df[df["gender"] == "Female"][c].mean() for c in score_cols]

    x  = np.arange(len(labels))
    w  = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Gender Analysis", fontsize=15, fontweight="bold")

    # Grouped bar
    b1 = ax1.bar(x - w/2, male_means,   w, label="Male",   color="#4C72B0", edgecolor="white")
    b2 = ax1.bar(x + w/2, female_means, w, label="Female", color="#DD8452", edgecolor="white")
    for bar in [*b1, *b2]:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean Score")
    ax1.set_title("Mean Scores by Gender")
    ax1.legend()
    ax1.set_ylim(0, 100)

    # Box plot
    male_data   = df[df["gender"] == "Male"]["average_score"]
    female_data = df[df["gender"] == "Female"]["average_score"]
    bp = ax2.boxplot(
        [male_data, female_data],
        labels=["Male", "Female"],
        patch_artist=True,
        notch=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#DD8452")
    ax2.set_ylabel("Average Score")
    ax2.set_title("Average Score Distribution by Gender")

    plt.tight_layout()
    _save(fig, "03_gender_analysis.png")


# ── 4. Study Hours vs Average Score ──────────────────────────────────────────
def plot_study_hours(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Study Hours vs Academic Performance", fontsize=15, fontweight="bold")

    # Scatter with regression line
    x = df["study_hours_per_day"]
    y = df["average_score"]
    ax1.scatter(x, y, alpha=0.45, c="#4C72B0", s=25, edgecolors="none")

    m, b  = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 200)
    ax1.plot(xline, m * xline + b, color="crimson", linewidth=2, label=f"y = {m:.2f}x + {b:.1f}")
    ax1.set_xlabel("Study Hours per Day")
    ax1.set_ylabel("Average Score")
    ax1.set_title("Scatter Plot + Regression Line")
    ax1.legend()

    # Bar by category
    if "study_category" in df.columns:
        grp = df.groupby("study_category", observed=True)["average_score"].mean()
        colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]
        bars = ax2.bar(range(len(grp)), grp.values, color=colors, edgecolor="white")
        for bar, val in zip(bars, grp.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        ax2.set_xticks(range(len(grp)))
        ax2.set_xticklabels(grp.index, rotation=20, ha="right", fontsize=9)
        ax2.set_ylabel("Mean Average Score")
        ax2.set_title("Mean Score by Study Category")

    plt.tight_layout()
    _save(fig, "04_study_hours_analysis.png")


# ── 5. Attendance Rate vs Average Score ──────────────────────────────────────
def plot_attendance(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Attendance Rate vs Academic Performance", fontsize=15, fontweight="bold")

    x = df["attendance_rate"]
    y = df["average_score"]
    ax1.scatter(x, y, alpha=0.45, c="#55A868", s=25, edgecolors="none")
    m, b  = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 200)
    ax1.plot(xline, m * xline + b, color="crimson", linewidth=2, label=f"y = {m:.2f}x + {b:.1f}")
    ax1.set_xlabel("Attendance Rate (%)")
    ax1.set_ylabel("Average Score")
    ax1.set_title("Scatter Plot + Regression Line")
    ax1.legend()

    if "attendance_category" in df.columns:
        grp    = df.groupby("attendance_category", observed=True)["average_score"].mean()
        colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]
        bars   = ax2.bar(range(len(grp)), grp.values, color=colors, edgecolor="white")
        for bar, val in zip(bars, grp.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        ax2.set_xticks(range(len(grp)))
        ax2.set_xticklabels(grp.index, rotation=20, ha="right", fontsize=9)
        ax2.set_ylabel("Mean Average Score")
        ax2.set_title("Mean Score by Attendance Category")

    plt.tight_layout()
    _save(fig, "05_attendance_analysis.png")


# ── 6. Sleep Hours Analysis ───────────────────────────────────────────────────
def plot_sleep_hours(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Sleep Hours vs Academic Performance", fontsize=15, fontweight="bold")

    x = df["sleep_hours"]
    y = df["average_score"]
    ax1.scatter(x, y, alpha=0.45, c="#8172B2", s=25, edgecolors="none")
    m, b  = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 200)
    ax1.plot(xline, m * xline + b, color="crimson", linewidth=2)
    ax1.set_xlabel("Sleep Hours per Night")
    ax1.set_ylabel("Average Score")
    ax1.set_title("Scatter Plot + Regression Line")

    if "sleep_category" in df.columns:
        grp    = df.groupby("sleep_category", observed=True)["average_score"].mean()
        colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]
        bars   = ax2.bar(range(len(grp)), grp.values, color=colors, edgecolor="white")
        for bar, val in zip(bars, grp.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        ax2.set_xticks(range(len(grp)))
        ax2.set_xticklabels(grp.index, rotation=20, ha="right", fontsize=9)
        ax2.set_ylabel("Mean Average Score")
        ax2.set_title("Mean Score by Sleep Category")

    plt.tight_layout()
    _save(fig, "06_sleep_analysis.png")


# ── 7. Parental Education Impact ─────────────────────────────────────────────
def plot_parental_education(df: pd.DataFrame) -> None:
    edu_order = [
        "No Formal Education", "Primary School", "High School",
        "Bachelor's Degree", "Master's Degree",
    ]
    grp   = df.groupby("parental_education", observed=True)["average_score"].mean().reindex(edu_order)
    grp2  = df.groupby("parental_education", observed=True)["average_score"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Parental Education Impact on Student Scores", fontsize=15, fontweight="bold")

    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(edu_order)))
    bars = ax1.barh(range(len(grp)), grp.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, grp.values):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va="center", fontsize=9)
    ax1.set_yticks(range(len(grp)))
    ax1.set_yticklabels(edu_order, fontsize=9)
    ax1.set_xlabel("Mean Average Score")
    ax1.set_title("Mean Score by Parental Education")
    ax1.set_xlim(0, 100)

    # Box plot
    data = [df[df["parental_education"] == edu]["average_score"].dropna().values for edu in edu_order]
    bp = ax2.boxplot(data, vert=True, patch_artist=True, labels=[e.replace(" ", "\n") for e in edu_order])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel("Average Score")
    ax2.set_title("Score Distribution by Parental Education")
    ax2.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    _save(fig, "07_parental_education.png")


# ── 8. Extracurricular & Internet ─────────────────────────────────────────────
def plot_extra_internet(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Extracurricular Activities & Internet Access Impact", fontsize=14, fontweight="bold")

    for ax, col, title, mapping in zip(
        axes,
        ["extracurricular", "internet_access"],
        ["Extracurricular Activities", "Internet Access"],
        [{0: "No", 1: "Yes"}, {0: "No", 1: "Yes"}],
    ):
        groups = {mapping[k]: df[df[col] == k]["average_score"].dropna() for k in df[col].unique()}
        bp = ax.boxplot(
            [groups[lbl] for lbl in sorted(groups)],
            labels=sorted(groups),
            patch_artist=True,
            notch=True,
            medianprops={"color": "black", "linewidth": 2},
        )
        colors = ["#DD8452", "#4C72B0"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        means = [groups[lbl].mean() for lbl in sorted(groups)]
        for i, mean in enumerate(means, start=1):
            ax.plot(i, mean, marker="D", color="red", markersize=7, zorder=5)
            ax.text(i + 0.07, mean, f"{mean:.1f}", va="center", fontsize=9)

        ax.set_title(title)
        ax.set_ylabel("Average Score")

    plt.tight_layout()
    _save(fig, "08_extracurricular_internet.png")


# ── 9. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    import matplotlib.colors as mcolors

    num_cols = [
        "study_hours_per_day", "attendance_rate", "sleep_hours",
        "internet_access", "extracurricular",
        "math_score", "reading_score", "writing_score", "average_score",
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Feature Correlation Heatmap", fontsize=14, fontweight="bold")

    cmap = plt.cm.RdYlGn
    im   = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Correlation Coefficient")

    n = len(num_cols)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [c.replace("_", "\n") for c in num_cols]
    ax.set_xticklabels(short, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(short, fontsize=8)

    for i in range(n):
        for j in range(n):
            val  = corr.values[i, j]
            text = ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=7, color="black" if abs(val) < 0.6 else "white")

    plt.tight_layout()
    _save(fig, "09_correlation_heatmap.png")


# ── 10. Score Comparison by Grade (Box Plots) ─────────────────────────────────
def plot_scores_by_grade(df: pd.DataFrame) -> None:
    grade_order = ["A", "B", "C", "D", "F"]
    score_cols  = ["math_score", "reading_score", "writing_score"]
    titles      = ["Math Score", "Reading Score", "Writing Score"]
    colors      = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Score Distribution by Grade", fontsize=14, fontweight="bold")

    for ax, col, title in zip(axes, score_cols, titles):
        data = [df[df["grade"] == g][col].dropna().values for g in grade_order]
        bp   = ax.boxplot(data, labels=grade_order, patch_artist=True,
                          medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(title)
        ax.set_xlabel("Grade")
        if ax == axes[0]:
            ax.set_ylabel("Score")

    plt.tight_layout()
    _save(fig, "10_scores_by_grade.png")


# ── 11. Dashboard Summary (combined) ─────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Student Performance Analysis — Dashboard", fontsize=18, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # (a) Average score histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["average_score"], bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax1.axvline(df["average_score"].mean(), color="crimson", linestyle="--", linewidth=1.5)
    ax1.set_title("Avg Score Distribution")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Students")

    # (b) Grade pie
    ax2 = fig.add_subplot(gs[0, 1])
    grade_order = ["A", "B", "C", "D", "F"]
    counts = df["grade"].value_counts().reindex(grade_order).fillna(0)
    ax2.pie(counts.values, labels=grade_order, autopct="%1.0f%%", startangle=140,
            colors=["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"],
            wedgeprops={"edgecolor": "white"})
    ax2.set_title("Grade Distribution")

    # (c) Gender bar
    ax3 = fig.add_subplot(gs[0, 2])
    gender_means = df.groupby("gender")["average_score"].mean()
    ax3.bar(gender_means.index, gender_means.values, color=["#4C72B0", "#DD8452"], edgecolor="white")
    ax3.set_title("Avg Score by Gender")
    ax3.set_ylabel("Score")
    ax3.set_ylim(0, 100)

    # (d) Study hours scatter
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(df["study_hours_per_day"], df["average_score"], alpha=0.4, s=15, c="#55A868")
    m, b = np.polyfit(df["study_hours_per_day"], df["average_score"], 1)
    xv = np.linspace(df["study_hours_per_day"].min(), df["study_hours_per_day"].max(), 100)
    ax4.plot(xv, m * xv + b, color="crimson", lw=1.5)
    ax4.set_title("Study Hours vs Score")
    ax4.set_xlabel("Hours/day")
    ax4.set_ylabel("Avg Score")

    # (e) Attendance scatter
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(df["attendance_rate"], df["average_score"], alpha=0.4, s=15, c="#C44E52")
    m, b = np.polyfit(df["attendance_rate"], df["average_score"], 1)
    xv = np.linspace(df["attendance_rate"].min(), df["attendance_rate"].max(), 100)
    ax5.plot(xv, m * xv + b, color="navy", lw=1.5)
    ax5.set_title("Attendance vs Score")
    ax5.set_xlabel("Attendance (%)")
    ax5.set_ylabel("Avg Score")

    # (f) Parental education bar
    ax6 = fig.add_subplot(gs[1, 2])
    edu_order  = ["No Formal Education", "Primary School", "High School", "Bachelor's Degree", "Master's Degree"]
    short_edu  = ["None", "Primary", "High Sch", "Bachelor", "Master"]
    edu_means  = df.groupby("parental_education", observed=True)["average_score"].mean().reindex(edu_order)
    ax6.bar(short_edu, edu_means.values, color=plt.cm.Blues(np.linspace(0.35, 0.85, 5)), edgecolor="white")
    ax6.set_title("Parental Education Impact")
    ax6.set_ylabel("Avg Score")
    ax6.tick_params(axis="x", labelsize=7, rotation=15)

    # (g) Extracurricular box plot
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.boxplot([df[df["extracurricular"] == 0]["average_score"],
                 df[df["extracurricular"] == 1]["average_score"]],
                labels=["No", "Yes"], patch_artist=True,
                medianprops={"color": "black", "lw": 2})
    ax7.set_title("Extracurricular vs Score")
    ax7.set_ylabel("Avg Score")

    # (h) Internet access box plot
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.boxplot([df[df["internet_access"] == 0]["average_score"],
                 df[df["internet_access"] == 1]["average_score"]],
                labels=["No", "Yes"], patch_artist=True,
                medianprops={"color": "black", "lw": 2})
    ax8.set_title("Internet Access vs Score")
    ax8.set_ylabel("Avg Score")

    # (i) Sleep hours scatter
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.scatter(df["sleep_hours"], df["average_score"], alpha=0.4, s=15, c="#8172B2")
    m, b = np.polyfit(df["sleep_hours"], df["average_score"], 1)
    xv = np.linspace(df["sleep_hours"].min(), df["sleep_hours"].max(), 100)
    ax9.plot(xv, m * xv + b, color="crimson", lw=1.5)
    ax9.set_title("Sleep Hours vs Score")
    ax9.set_xlabel("Sleep Hours")
    ax9.set_ylabel("Avg Score")

    plt.tight_layout()
    _save(fig, "00_dashboard.png")
