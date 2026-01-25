"""
data_ingestion.py
-----------------
Loads, cleans, and converts each healthcare CSV dataset
into LangChain Document objects (one Document per row).
"""

import pandas as pd
from langchain_core.documents import Document


# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip column names, replace spaces with underscores."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _drop_high_null_cols(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns where more than `threshold` fraction of values are null."""
    null_frac = df.isnull().mean()
    keep = null_frac[null_frac <= threshold].index.tolist()
    return df[keep]


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with median, categorical with 'unknown'."""
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("unknown")
    return df


# ──────────────────────────────────────────────
# Dataset-specific row → text converters
# ──────────────────────────────────────────────

def _heart_row_to_text(row: pd.Series) -> str:
    """Convert a heart disease dataset row to a natural-language summary."""
    # Map numeric targets if present
    target_map = {0: "no heart disease", 1: "heart disease present"}
    target = target_map.get(int(row.get("target", -1)), "unknown")

    cp_map = {0: "typical angina", 1: "atypical angina",
              2: "non-anginal pain", 3: "asymptomatic"}
    cp = cp_map.get(int(row.get("cp", -1)), str(row.get("cp", "unknown")))

    sex = "Male" if int(row.get("sex", -1)) == 1 else "Female"

    parts = [
        f"Patient is a {int(row.get('age', 0))}-year-old {sex}.",
        f"Chest pain type: {cp}.",
        f"Resting blood pressure: {row.get('trestbps', 'N/A')} mmHg.",
        f"Cholesterol level: {row.get('chol', 'N/A')} mg/dl.",
        f"Fasting blood sugar > 120 mg/dl: {'Yes' if row.get('fbs', 0) == 1 else 'No'}.",
        f"Maximum heart rate achieved: {row.get('thalach', 'N/A')} bpm.",
        f"Exercise-induced angina: {'Yes' if row.get('exang', 0) == 1 else 'No'}.",
        f"ST depression (oldpeak): {row.get('oldpeak', 'N/A')}.",
        f"Diagnosis: {target}.",
    ]
    return " ".join(parts)


def _dermatology_row_to_text(row: pd.Series) -> str:
    """Convert a dermatology dataset row to a natural-language summary."""
    # The class column maps 1-6 to disease names
    class_map = {
        1: "Psoriasis",
        2: "Seboreic Dermatitis",
        3: "Lichen Planus",
        4: "Pityriasis Rosea",
        5: "Chronic Dermatitis",
        6: "Pityriasis Rubra Pilaris",
    }
    cls_raw = row.get("class", row.get("diagnosis", "unknown"))
    try:
        cls_name = class_map.get(int(cls_raw), str(cls_raw))
    except (ValueError, TypeError):
        cls_name = str(cls_raw)

    # Collect all feature columns except 'class'
    feature_cols = [c for c in row.index if c not in ("class", "diagnosis")]
    feature_parts = []
    for col in feature_cols[:12]:  # limit to keep text concise
        feature_parts.append(f"{col.replace('_', ' ')}: {row[col]}")

    parts = [
        f"Dermatology case with diagnosis: {cls_name}.",
        "Clinical features — " + ", ".join(feature_parts) + ".",
    ]
    return " ".join(parts)


def _diabetes_row_to_text(row: pd.Series) -> str:
    """Convert a Pakistani diabetes dataset row to a natural-language summary."""
    outcome_map = {0: "non-diabetic", 1: "diabetic"}
    outcome = outcome_map.get(int(row.get("outcome", row.get("diabetes", -1))), "unknown")

    parts = [
        f"Patient data — Age: {row.get('age', 'N/A')}.",
        f"BMI: {row.get('bmi', 'N/A')}.",
        f"Glucose level: {row.get('glucose', row.get('bloodglucose', 'N/A'))}.",
        f"Blood pressure: {row.get('bloodpressure', row.get('blood_pressure', 'N/A'))} mmHg.",
        f"Insulin level: {row.get('insulin', 'N/A')}.",
        f"Skin thickness: {row.get('skinthickness', row.get('skin_thickness', 'N/A'))} mm.",
        f"Diabetes pedigree function: {row.get('diabetespedigreefunction', row.get('diabetes_pedigree_function', 'N/A'))}.",
        f"Pregnancies: {row.get('pregnancies', 'N/A')}.",
        f"Outcome: {outcome}.",
    ]
    return " ".join(parts)


# ──────────────────────────────────────────────
# Public loading functions
# ──────────────────────────────────────────────

def load_heart_disease(csv_path: str) -> list[Document]:
    """Load and convert the heart disease CSV into Documents."""
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    df = _drop_high_null_cols(df)
    df = _fill_missing(df)

    docs = []
    for idx, row in df.iterrows():
        text = _heart_row_to_text(row)
        metadata = {
            "dataset_name": "heart_disease",
            "row_index": int(idx),
            "age": str(row.get("age", "")),
            "sex": str(row.get("sex", "")),
            "target": str(row.get("target", "")),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def load_dermatology(csv_path: str) -> list[Document]:
    """Load and convert the dermatology CSV into Documents."""
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    df = _drop_high_null_cols(df)
    df = _fill_missing(df)

    docs = []
    for idx, row in df.iterrows():
        text = _dermatology_row_to_text(row)
        metadata = {
            "dataset_name": "dermatology",
            "row_index": int(idx),
            "diagnosis": str(row.get("class", row.get("diagnosis", ""))),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def load_diabetes(csv_path: str) -> list[Document]:
    """Load and convert the Pakistani diabetes CSV into Documents."""
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    df = _drop_high_null_cols(df)
    df = _fill_missing(df)

    docs = []
    for idx, row in df.iterrows():
        text = _diabetes_row_to_text(row)
        metadata = {
            "dataset_name": "diabetes",
            "row_index": int(idx),
            "outcome": str(row.get("outcome", row.get("diabetes", ""))),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs
