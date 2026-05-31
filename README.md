# Student Performance Analysis

A complete Python data analysis project that explores factors affecting student academic performance using Pandas and Matplotlib.

## Project Structure

```
Student-Performance-analysis/
├── data/
│   └── student_data.csv         # Generated dataset (500 students)
├── plots/                        # Output charts (auto-created)
├── src/
│   ├── preprocess.py            # Data loading, cleaning, feature engineering
│   ├── eda.py                   # Exploratory data analysis & statistics
│   └── visualizations.py        # All matplotlib chart functions
├── generate_data.py             # Synthetic dataset generator
├── analysis.py                  # Main analysis pipeline (entry point)
└── requirements.txt
```

## Dataset Features

| Column | Description |
|---|---|
| student_id | Unique student identifier |
| gender | Male / Female |
| age | 14 – 18 |
| parental_education | No Formal Education → Master's Degree |
| internet_access | 0 = No, 1 = Yes |
| extracurricular | 0 = No, 1 = Yes |
| study_hours_per_day | 0.5 – 8.0 hours |
| attendance_rate | 55 – 100 % |
| sleep_hours | 4.0 – 10.0 hours |
| math_score | 0 – 100 |
| reading_score | 0 – 100 |
| writing_score | 0 – 100 |
| average_score | Mean of three subject scores |
| grade | A / B / C / D / F |

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python generate_data.py

# 3. Run full analysis + generate charts
python analysis.py

# 4. Run without charts
python analysis.py --no-plots

# 5. Use a custom dataset
python analysis.py --data path/to/your_data.csv
```

## Charts Generated (plots/)

| File | Description |
|---|---|
| 00_dashboard.png | 9-panel summary dashboard |
| 01_score_distributions.png | Histograms of all four scores |
| 02_grade_distribution.png | Bar + pie chart of grade counts |
| 03_gender_analysis.png | Grouped bars + box plot by gender |
| 04_study_hours_analysis.png | Scatter regression + bar by category |
| 05_attendance_analysis.png | Scatter regression + bar by category |
| 06_sleep_analysis.png | Scatter regression + bar by category |
| 07_parental_education.png | Horizontal bar + box plot by edu level |
| 08_extracurricular_internet.png | Box plots for binary factors |
| 09_correlation_heatmap.png | Pearson correlation matrix heatmap |
| 10_scores_by_grade.png | Subject score box plots grouped by grade |

## Tools Used

- **Python 3.8+**
- **Pandas** — data loading, cleaning, aggregation
- **NumPy** — numerical operations & synthetic data generation
- **Matplotlib** — all visualizations
