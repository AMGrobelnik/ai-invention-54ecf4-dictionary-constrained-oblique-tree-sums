# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0.0",
# ]
# ///
"""
data.py — Tabular Benchmark Dataset Standardizer for DOTS

Loads the imodels/tabular-benchmark-797-classification dataset (selected as the
BEST match for the DOTS benchmark suite) and outputs 200 standardized examples
in exp_sel_data_out.json format.

Selection rationale: This dataset is directly from the OpenML-797 benchmark suite
used in RO-FIGS and related interpretable ML papers. It has 44 pure numeric features
(F1R..F22R, F1S..F22S), binary classification target, 3000 samples, and requires
no preprocessing — ideal for DOTS dictionary-constrained oblique tree evaluation.

Each example represents a tabular data row as a classification task:
- input: Feature vector as a structured text description
- context: Full feature dictionary + metadata
- output: Binary classification label (0 or 1)
- dataset: Source dataset name
- split: Original split name
"""

import json
import random
from pathlib import Path
from typing import Any

# AIDEV-NOTE: Working directory is the script's parent directory
WD = Path(__file__).parent
DATASETS_DIR = WD / "temp" / "datasets"
OUTPUT_FILE = WD / "data_out.json"
EXAMPLES_PER_DATASET = 200
RANDOM_SEED = 42


def load_json(filepath: Path) -> list[dict[str, Any]]:
    """Load a JSON file and return list of records."""
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_feature_name(name: str) -> str:
    """Clean feature name to valid Python identifier."""
    cleaned = name.replace(" ", "_").replace("-", "_").replace(".", "_")
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in cleaned)
    if cleaned and cleaned[0].isdigit():
        cleaned = "f_" + cleaned
    return cleaned


def format_features_as_text(features: dict[str, Any]) -> str:
    """Format feature dict as a readable text description for the input field."""
    lines = []
    for k, v in features.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")
    return "Classify the following tabular data sample:\n" + "\n".join(lines)


def process_tabular_benchmark(
    records: list[dict[str, Any]],
    n_examples: int,
) -> list[dict[str, Any]]:
    """Process imodels/tabular-benchmark-797-classification dataset.

    This dataset has 44 numeric features (F1R..F22R, F1S..F22S) + binary target.
    Already fully numeric — no preprocessing needed beyond field standardization.
    """
    # AIDEV-NOTE: Remove index columns that are not features
    drop_cols = {"Unnamed: 0.1", "Unnamed: 0", "target"}
    feature_cols = [c for c in records[0].keys() if c not in drop_cols]
    feature_cols_clean = {c: clean_feature_name(c) for c in feature_cols}

    # Sample n_examples deterministically
    rng = random.Random(RANDOM_SEED)
    if len(records) > n_examples:
        sampled = rng.sample(records, n_examples)
    else:
        sampled = records

    examples = []
    for row in sampled:
        target = int(row["target"])
        features = {
            feature_cols_clean[c]: row[c] for c in feature_cols
        }

        example = {
            "input": format_features_as_text(features),
            "context": {
                "features": features,
                "n_features": len(features),
                "task_type": "binary_classification",
                "dataset_source": "OpenML-797 benchmark suite",
                "feature_type": "numeric",
                "preprocessing": "none_needed_already_numeric",
            },
            "output": str(target),
            "dataset": "imodels/tabular-benchmark-797-classification",
            "split": "test",
        }
        examples.append(example)

    return examples


def process_churn_prediction(
    records: list[dict[str, Any]],
    n_examples: int,
) -> list[dict[str, Any]]:
    """Process scikit-learn/churn-prediction dataset.

    This dataset has mixed features (categorical + numeric) + binary Churn target.
    Preprocessing: one-hot encode categoricals, convert target to 0/1.
    """
    # AIDEV-NOTE: Drop customerID (identifier, not a feature)
    id_cols = {"customerID"}
    target_col = "Churn"

    # Identify categorical vs numeric features
    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ]
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

    # Encode target: Yes=1, No=0
    target_map = {"Yes": 1, "No": 0}

    # Sample n_examples deterministically
    rng = random.Random(RANDOM_SEED)
    if len(records) > n_examples:
        sampled = rng.sample(records, n_examples)
    else:
        sampled = records

    examples = []
    for row in sampled:
        target_val = row.get(target_col)
        if target_val is None:
            continue
        target = target_map.get(str(target_val), int(target_val) if str(target_val).isdigit() else 0)

        # Build feature dict with one-hot encoding for categoricals
        features: dict[str, Any] = {}

        # Numeric features
        for col in numeric_cols:
            val = row.get(col)
            if val is None or val == "" or val == " ":
                # AIDEV-NOTE: Impute missing with 0.0 (median would require full dataset scan)
                features[clean_feature_name(col)] = 0.0
            else:
                try:
                    features[clean_feature_name(col)] = float(val)
                except (ValueError, TypeError):
                    features[clean_feature_name(col)] = 0.0

        # One-hot encode categorical features
        for col in categorical_cols:
            val = str(row.get(col, "Unknown"))
            ohe_name = clean_feature_name(f"{col}_{val}")
            # AIDEV-NOTE: Set current category to 1, others implicitly 0 in context
            features[ohe_name] = 1

        # Build original features dict (before encoding) for context
        original_features = {}
        for col in numeric_cols + categorical_cols:
            if col not in id_cols:
                original_features[col] = row.get(col)

        example = {
            "input": format_features_as_text(features),
            "context": {
                "features": features,
                "original_features": original_features,
                "n_features_original": len(numeric_cols) + len(categorical_cols),
                "n_features_encoded": len(features),
                "task_type": "binary_classification",
                "dataset_source": "scikit-learn/churn-prediction (IBM Telco)",
                "feature_type": "mixed_categorical_numeric",
                "preprocessing": "one_hot_encoded_categoricals",
                "target_encoding": {"Yes": 1, "No": 0},
            },
            "output": str(target),
            "dataset": "scikit-learn/churn-prediction",
            "split": "train",
        }
        examples.append(example)

    return examples


def main() -> None:
    """Main processing pipeline.

    AIDEV-NOTE: Only outputs the BEST dataset (tabular-benchmark-797-classification).
    Selected over churn-prediction because it's directly from the OpenML-797 benchmark
    suite used in RO-FIGS, has pure numeric features, and needs no preprocessing.
    """
    print(f"Loading datasets from: {DATASETS_DIR}")

    # Load the selected best dataset
    tabular_file = DATASETS_DIR / "full_imodels_tabular-benchmark-797-classification_test.json"
    tabular_records = load_json(tabular_file)
    print(f"  tabular-benchmark-797: {len(tabular_records)} records loaded")

    # Process selected dataset (200 examples)
    tabular_examples = process_tabular_benchmark(
        records=tabular_records,
        n_examples=EXAMPLES_PER_DATASET,
    )
    print(f"  tabular-benchmark-797: {len(tabular_examples)} examples extracted")

    # Build output matching exp_sel_data_out.json schema
    output = {"examples": tabular_examples}

    # Save to full_data_out.json
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"Total examples: {len(tabular_examples)}")

    # Summary statistics
    targets = [int(e["output"]) for e in tabular_examples]
    class_1_frac = sum(targets) / len(targets) if targets else 0
    print(f"\n  tabular-benchmark-797-classification (SELECTED):")
    print(f"    Source: imodels/tabular-benchmark-797-classification (OpenML-797 suite)")
    print(f"    Task: binary classification")
    print(f"    Features: {tabular_examples[0]['context']['n_features']} (all numeric)")
    print(f"    Class balance: {class_1_frac:.3f} (class=1)")
    print(f"    Examples: {len(tabular_examples)}")


if __name__ == "__main__":
    main()
