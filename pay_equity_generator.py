"""
Project 5: Pay Equity Analysis Dashboard
Data Generator — Synthetic HR Compensation Dataset

Research foundation:
    Mercer Global Pay Equity Report: unadjusted gender pay gap ~18% globally;
    adjusted gap narrows to 2-3% after controlling for grade, function, and location.

    Willis Towers Watson Pay Equity Audit Framework: compa-ratio (actual / midpoint)
    is the primary equity metric; target band 0.85-1.15.

    WTW finding: approximately 66% of total pay inequity is explained by grade
    distribution differences, not direct pay decisions within the same role.

Methodology applied:
    1. Grade distribution tables are calibrated by demographic group to replicate
       the structural clustering effect documented in WTW research.
    2. A direct within-grade salary adjustment of 2-3% encodes the residual gap
       identified in Mercer's adjusted pay gap analysis.
    3. The combination produces a realistic unadjusted gap of approximately 14-17%
       that narrows to 2-3% when grade and department are held constant.

Dataset: 1,000 synthetic employees
Output:  pay_equity_data.csv, pay_equity_data.xlsx
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    from faker import Faker
    fake = Faker("en_CA")
    Faker.seed(42)
    USE_FAKER = True
except ImportError:
    USE_FAKER = False

np.random.seed(42)
random.seed(42)

N = 1000
REFERENCE_DATE = datetime(2024, 6, 1)

# ---------------------------------------------------------------------------
# Job structure reference tables
# ---------------------------------------------------------------------------

DEPARTMENTS = {
    "Engineering": [
        "Software Engineer",
        "Data Engineer",
        "DevOps Engineer",
        "QA Engineer",
    ],
    "Finance": [
        "Financial Analyst",
        "Accountant",
        "FP&A Analyst",
        "Treasury Analyst",
    ],
    "Human Resources": [
        "HR Generalist",
        "Talent Acquisition Specialist",
        "Compensation Analyst",
        "HR Business Partner",
    ],
    "Marketing": [
        "Marketing Analyst",
        "Brand Manager",
        "Content Strategist",
        "Digital Marketer",
    ],
    "Operations": [
        "Operations Analyst",
        "Supply Chain Analyst",
        "Process Improvement Specialist",
        "Logistics Coordinator",
    ],
    "Legal": [
        "Legal Counsel",
        "Paralegal",
        "Compliance Officer",
        "Contract Specialist",
    ],
    "Sales": [
        "Account Executive",
        "Sales Analyst",
        "Business Development Rep",
        "Customer Success Manager",
    ],
    "Product": [
        "Product Manager",
        "Product Analyst",
        "UX Researcher",
        "Technical Program Manager",
    ],
}

# Pay bands in CAD, aligned to Mercer Total Remuneration Survey benchmarks
GRADES = {
    "G1": {"level": "Entry",          "min": 45_000,  "mid": 55_000,  "max": 65_000},
    "G2": {"level": "Junior",         "min": 60_000,  "mid": 72_000,  "max": 85_000},
    "G3": {"level": "Mid-Level",      "min": 80_000,  "mid": 95_000,  "max": 110_000},
    "G4": {"level": "Senior",         "min": 100_000, "mid": 120_000, "max": 140_000},
    "G5": {"level": "Lead / Manager", "min": 130_000, "mid": 152_000, "max": 175_000},
    "G6": {"level": "Director+",      "min": 160_000, "mid": 190_000, "max": 225_000},
}

GRADE_KEYS = list(GRADES.keys())

# ---------------------------------------------------------------------------
# Demographic options and probabilities
# ---------------------------------------------------------------------------

GENDER_OPTIONS = ["Male", "Female", "Non-binary"]
GENDER_PROBS   = [0.52, 0.44, 0.04]

ETHNICITY_OPTIONS = [
    "White",
    "Black or African American",
    "Hispanic or Latino",
    "Asian",
    "Two or more races",
    "Other / prefer not to say",
]
ETHNICITY_PROBS = [0.55, 0.12, 0.14, 0.12, 0.04, 0.03]

LOCATIONS       = ["Toronto", "Vancouver", "Calgary", "Ottawa", "Montreal", "Remote"]
LOCATION_PROBS  = [0.35, 0.20, 0.15, 0.12, 0.10, 0.08]

PERFORMANCE_RATINGS = [1, 2, 3, 4, 5]
PERFORMANCE_PROBS   = [0.03, 0.12, 0.50, 0.28, 0.07]

# ---------------------------------------------------------------------------
# Grade distribution by demographic group
#
# This is the core mechanism that creates the unadjusted pay gap.
# Women and underrepresented minority employees are assigned lower grades
# at higher rates, replicating the structural clustering WTW documents as
# the dominant driver of total compensation inequality.
# ---------------------------------------------------------------------------

GRADE_DIST = {
    "default":    [0.08, 0.15, 0.28, 0.25, 0.15, 0.09],  # White or Asian male
    "female":     [0.14, 0.22, 0.30, 0.19, 0.10, 0.05],  # Overrepresented in G1-G3
    "non_binary": [0.13, 0.21, 0.30, 0.20, 0.11, 0.05],
    "urm":        [0.12, 0.20, 0.29, 0.21, 0.12, 0.06],  # Black or Hispanic / Latino
}

URM_ETHNICITIES = {"Black or African American", "Hispanic or Latino"}


def get_grade_weights(gender: str, ethnicity: str) -> list:
    """Return the grade probability distribution for a demographic combination."""
    if gender == "Female":
        return GRADE_DIST["female"]
    if gender == "Non-binary":
        return GRADE_DIST["non_binary"]
    if ethnicity in URM_ETHNICITIES:
        return GRADE_DIST["urm"]
    return GRADE_DIST["default"]


# ---------------------------------------------------------------------------
# Salary generation
#
# Two-layer pay gap:
#   Layer 1 (structural) — grade clustering above creates the raw gap
#   Layer 2 (direct)     — a within-grade multiplier of 2-3% encodes the
#                          residual discrimination found in Mercer adjusted analysis
# ---------------------------------------------------------------------------

def generate_salary(
    grade: str,
    gender: str,
    ethnicity: str,
    perf_rating: int,
) -> int:
    """
    Generate a base salary within the grade band, applying demographic
    adjustments to replicate real-world pay equity patterns.
    """
    g = GRADES[grade]

    # Draw from a uniform within the pay band
    salary = np.random.uniform(g["min"], g["max"])

    # Direct within-grade gap — encodes Mercer adjusted gap of ~2-3%
    if gender == "Female":
        salary *= np.random.uniform(0.965, 0.990)
    elif gender == "Non-binary":
        salary *= np.random.uniform(0.960, 0.990)

    if ethnicity in URM_ETHNICITIES:
        salary *= np.random.uniform(0.972, 0.995)

    # Performance position: higher performers sit higher within their band
    perf_nudge = 1.0 + (perf_rating - 3) * 0.015
    salary *= perf_nudge

    # Clamp to band limits
    salary = float(np.clip(salary, g["min"], g["max"]))
    return int(round(salary, 0))


# ---------------------------------------------------------------------------
# Date and experience helpers
# ---------------------------------------------------------------------------

def random_hire_date() -> datetime:
    start = datetime(2010, 1, 1)
    end   = datetime(2024, 1, 1)
    return start + timedelta(days=random.randint(0, (end - start).days))


def calc_years_tenure(hire_date: datetime) -> float:
    return round((REFERENCE_DATE - hire_date).days / 365.25, 1)


def calc_years_experience(hire_date: datetime, age: int) -> float:
    tenure = calc_years_tenure(hire_date)
    prior  = max(0.0, age - 22.0 - tenure)
    return round(tenure + prior, 1)


# ---------------------------------------------------------------------------
# Main dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(n: int = N) -> pd.DataFrame:
    """
    Generate n synthetic employee compensation records with realistic
    pay equity patterns calibrated to Mercer and WTW benchmarks.
    """
    records = []

    for i in range(n):
        # --- Demographics ---
        gender    = str(np.random.choice(GENDER_OPTIONS, p=GENDER_PROBS))
        ethnicity = str(np.random.choice(ETHNICITY_OPTIONS, p=ETHNICITY_PROBS))
        age       = int(np.clip(np.random.normal(38, 9), 22, 62))

        # --- Job structure ---
        dept      = random.choice(list(DEPARTMENTS.keys()))
        job_title = random.choice(DEPARTMENTS[dept])
        grade     = str(np.random.choice(GRADE_KEYS, p=get_grade_weights(gender, ethnicity)))
        grade_info = GRADES[grade]

        # --- Performance ---
        perf_rating = int(np.random.choice(PERFORMANCE_RATINGS, p=PERFORMANCE_PROBS))

        # --- Dates ---
        hire_date        = random_hire_date()
        years_tenure     = calc_years_tenure(hire_date)
        years_experience = calc_years_experience(hire_date, age)

        # --- Compensation ---
        base_salary = generate_salary(grade, gender, ethnicity, perf_rating)
        range_min   = grade_info["min"]
        range_mid   = grade_info["mid"]
        range_max   = grade_info["max"]

        compa_ratio       = round(base_salary / range_mid, 4)
        range_penetration = round(
            (base_salary - range_min) / (range_max - range_min), 4
        )
        in_band_flag = 1 if range_min <= base_salary <= range_max else 0

        # Bonus — scales with performance rating
        bonus_pct    = round(
            np.random.uniform(0.04, 0.20) * (1 + (perf_rating - 3) * 0.02), 4
        )
        bonus_amount = int(round(base_salary * bonus_pct, 0))
        total_comp   = base_salary + bonus_amount

        # Last merit increase — performance-weighted
        last_increase_pct = round(
            np.random.uniform(0.02, 0.08) * (1 + (perf_rating - 3) * 0.008) * 100, 2
        )

        # Promotion flag
        promoted_last_2yr = int(np.random.choice([0, 1], p=[0.75, 0.25]))

        # Location
        location  = str(np.random.choice(LOCATIONS, p=LOCATION_PROBS))
        is_remote = 1 if location == "Remote" else 0

        records.append(
            {
                "employee_id":        f"EMP{str(i + 1001).zfill(5)}",
                "gender":             gender,
                "ethnicity":          ethnicity,
                "age":                age,
                "location":           location,
                "is_remote":          is_remote,
                "department":         dept,
                "job_title":          job_title,
                "job_grade":          grade,
                "job_level":          grade_info["level"],
                "hire_date":          hire_date.strftime("%Y-%m-%d"),
                "years_tenure":       years_tenure,
                "years_experience":   years_experience,
                "performance_rating": perf_rating,
                "promoted_last_2yr":  promoted_last_2yr,
                "base_salary":        base_salary,
                "range_min":          range_min,
                "range_mid":          range_mid,
                "range_max":          range_max,
                "compa_ratio":        compa_ratio,
                "range_penetration":  range_penetration,
                "in_band_flag":       in_band_flag,
                "bonus_pct":          bonus_pct,
                "bonus_amount":       bonus_amount,
                "total_comp":         total_comp,
                "last_increase_pct":  last_increase_pct,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Validation summary — printed on generation
# ---------------------------------------------------------------------------

def print_validation(df: pd.DataFrame) -> None:
    """Print a validation summary to verify the dataset matches expected benchmarks."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal employees : {len(df):,}")
    print(f"Columns         : {len(df.columns)}")

    print("\n--- Gender distribution ---")
    print(df["gender"].value_counts(normalize=True).mul(100).round(1).to_string())

    print("\n--- Average base salary by gender ---")
    gender_pay = df.groupby("gender")["base_salary"].mean()
    male_avg   = gender_pay.get("Male", float("nan"))
    female_avg = gender_pay.get("Female", float("nan"))
    if male_avg and female_avg:
        unadjusted_gap = round((male_avg - female_avg) / male_avg * 100, 1)
        print(f"  Male avg        : CAD {male_avg:,.0f}")
        print(f"  Female avg      : CAD {female_avg:,.0f}")
        print(f"  Unadjusted gap  : {unadjusted_gap}%  (Mercer benchmark ~18%)")

    print("\n--- Average compa-ratio by gender ---")
    print(df.groupby("gender")["compa_ratio"].mean().round(4).to_string())

    print("\n--- Grade distribution by gender (headcount) ---")
    grade_gender = (
        df.groupby(["job_grade", "gender"])
        .size()
        .unstack(fill_value=0)
    )
    print(grade_gender.to_string())

    print("\n--- Pay band compliance ---")
    in_band_pct = df["in_band_flag"].mean() * 100
    print(f"  Employees within pay band : {in_band_pct:.1f}%")

    print("\n--- Compa-ratio below 0.90 by gender ---")
    below_90 = (
        df[df["compa_ratio"] < 0.90]
        .groupby("gender")
        .size()
        .rename("count_below_0.90")
    )
    print(below_90.to_string())

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Export the dataset to CSV and Excel."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path  = os.path.join(output_dir, "pay_equity_data.csv")
    xlsx_path = os.path.join(output_dir, "pay_equity_data.xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, sheet_name="PayEquityData")

    print(f"\nOutput files:")
    print(f"  CSV   : {csv_path}")
    print(f"  Excel : {xlsx_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating pay equity dataset ...")
    df = generate_dataset(N)
    print_validation(df)
    export(df)
    print("\nDone.")
