"""Run a quick correlation-based screen for residential hedonic features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from hedonic.common.modeling_common import (
    LOC_NEIGHBORHOOD_FEATURES,
    STRUCTURAL_FEATURES,
    TARGET_COL,
    available_features,
    infer_feature_types,
    require_existing_path,
    subset_residential_rows,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parcel_path = require_existing_path(repo_root / "parcels_preprocessed.csv", "Parcel CSV")
    
    print(f"Reading: {parcel_path}")
    df = pd.read_csv(parcel_path, low_memory=False)
    before_filter = len(df)
    df = subset_residential_rows(df, strict=True)
    print(f"Residential subset rows: {len(df)} of {before_filter}")
    
    # Clean data
    df["TOTAL_VALUE_LOG"] = np.log1p(df[TARGET_COL])
    df = df[df[TARGET_COL] > 0]

    domain_features = available_features(df, [*STRUCTURAL_FEATURES, *LOC_NEIGHBORHOOD_FEATURES])
    _, categorical_features = infer_feature_types(df, domain_features)
    categorical_pool = set(categorical_features)
    
    # Define feature groups using only shared modeling_common variables.
    structural_only = STRUCTURAL_FEATURES
    locational_only = LOC_NEIGHBORHOOD_FEATURES
    structural_without_land = [
        feature for feature in STRUCTURAL_FEATURES if feature != "LAND_SF"
    ]
    size_only = [
        feature
        for feature in STRUCTURAL_FEATURES
        if feature in {"LAND_SF", "GROSS_AREA", "LIVING_AREA"}
    ]
    all_features = [*STRUCTURAL_FEATURES, *LOC_NEIGHBORHOOD_FEATURES]
    
    specs = [
        ("Structural only", structural_only),
        ("Locational/neighborhood only", locational_only),
        ("Structural + locational/neighborhood", all_features),
        ("Structural without LAND_SF", structural_without_land),
        ("Size only", size_only),
    ]
    
    print("\n" + "="*80)
    print("RESIDENTIAL FEATURE CORRELATION WITH LOG(TOTAL_VALUE)")
    print("="*80)
    
    results = []
    for label, features in specs:
        available = available_features(df, features)
        
        if not available:
            continue
        
        # Compute average absolute correlation of features with target
        correlations = []
        for feat in available:
            clean_data = df[[feat, "TOTAL_VALUE_LOG"]].copy()
            if feat in categorical_pool:
                clean_data[feat] = clean_data[feat].astype("string").str.strip()
                clean_data[feat] = clean_data[feat].replace(
                    {"": pd.NA, "nan": pd.NA, "None": pd.NA}
                )
                codes = clean_data[feat].astype("category").cat.codes
                clean_data[feat] = codes.replace(-1, np.nan)
            else:
                clean_data[feat] = pd.to_numeric(clean_data[feat], errors="coerce")
            clean_data["TOTAL_VALUE_LOG"] = pd.to_numeric(
                clean_data["TOTAL_VALUE_LOG"], errors="coerce"
            )
            clean_data = clean_data.dropna()
            if len(clean_data) > 10:
                corr = abs(clean_data[feat].corr(clean_data["TOTAL_VALUE_LOG"]))
                correlations.append(corr)
        
        avg_corr = np.mean(correlations) if correlations else 0
        
        results.append({
            "label": label,
            "n_features": len(available),
            "avg_correlation": avg_corr,
        })
        
        print(f"\n{label} ({len(available)} features)")
        print(f"  Avg |correlation| with log(TOTAL_VALUE): {avg_corr:.4f}")
        print(f"  Features: {', '.join(available)}")
    
    print("\n" + "="*80)
    print("SUMMARY (sorted by average correlation)")
    print("="*80)
    results_df = pd.DataFrame(results).sort_values("avg_correlation", ascending=False)
    for _, row in results_df.iterrows():
        print(f"{row['label']:40} | N: {row['n_features']:>2} | Avg |r|: {row['avg_correlation']:>6.4f}")
    
    print("\nNOTE: High average correlation suggests features that explain more value variation.")
    print("This is a fast proxy for model performance (correlation ≠ R² but generally correlated).")
    print("Categorical features are approximated via integer category codes for this quick screen.")

if __name__ == "__main__":
    main()
