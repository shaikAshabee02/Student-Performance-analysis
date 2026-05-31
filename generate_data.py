"""
Generates a realistic synthetic student performance dataset and saves it to data/student_data.csv.
Run this once before running analysis.py.
"""
import os
import random
import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

N = 500

PARENTAL_EDU = ["No Formal Education", "Primary School", "High School", "Bachelor's Degree", "Master's Degree"]
EDU_WEIGHT    = [0.05, 0.10, 0.35, 0.35, 0.15]

gender            = np.random.choice(["Male", "Female"], size=N, p=[0.50, 0.50])
age               = np.random.randint(14, 19, size=N)
parental_edu      = np.random.choice(PARENTAL_EDU, size=N, p=EDU_WEIGHT)
internet_access   = np.random.choice([0, 1], size=N, p=[0.15, 0.85])
extracurricular   = np.random.choice([0, 1], size=N, p=[0.40, 0.60])
study_hours       = np.round(np.random.uniform(0.5, 8.0, size=N), 1)
attendance_rate   = np.round(np.random.uniform(55, 100, size=N), 1)
sleep_hours       = np.round(np.random.uniform(4.0, 10.0, size=N), 1)

# Build a hidden "ability" score influenced by controllable factors
parental_boost = {
    "No Formal Education": -8,
    "Primary School":       -4,
    "High School":           0,
    "Bachelor's Degree":     5,
    "Master's Degree":      10,
}

ability = (
    0.0
    + np.vectorize(parental_boost.get)(parental_edu)
    + study_hours * 3.5
    + (attendance_rate - 75) * 0.4
    + (sleep_hours - 7) * 1.5
    + extracurricular * 2
    + internet_access * 3
    + np.random.normal(0, 8, size=N)     # random noise
)

def score_from_ability(ability_arr, mean_shift=0, noise_std=6):
    raw = 50 + ability_arr + mean_shift + np.random.normal(0, noise_std, size=N)
    return np.clip(np.round(raw, 0).astype(int), 0, 100)

math_score    = score_from_ability(ability,  mean_shift=0,   noise_std=7)
reading_score = score_from_ability(ability,  mean_shift=2,   noise_std=6)
writing_score = score_from_ability(ability,  mean_shift=1,   noise_std=6)

average_score = np.round((math_score + reading_score + writing_score) / 3, 1)

def assign_grade(avg):
    if avg >= 85: return "A"
    if avg >= 70: return "B"
    if avg >= 55: return "C"
    if avg >= 40: return "D"
    return "F"

grade = [assign_grade(s) for s in average_score]

df = pd.DataFrame({
    "student_id":          [f"S{str(i+1).zfill(3)}" for i in range(N)],
    "gender":              gender,
    "age":                 age,
    "parental_education":  parental_edu,
    "internet_access":     internet_access,
    "extracurricular":     extracurricular,
    "study_hours_per_day": study_hours,
    "attendance_rate":     attendance_rate,
    "sleep_hours":         sleep_hours,
    "math_score":          math_score,
    "reading_score":       reading_score,
    "writing_score":       writing_score,
    "average_score":       average_score,
    "grade":               grade,
})

os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "student_data.csv")
df.to_csv(out_path, index=False)
print(f"Dataset saved to: {out_path}  ({N} rows)")
print(df.head())
