#!/usr/bin/env python3
"""
main_phase2_simplified.py
Phase 2 pipeline WITHOUT Bayesian optimization (due to optuna install issues).

Instead, this runs Phase 2 with:
1. Nominal ODE parameters (no Bayesian search)
2. RF calibration enabled (CalibratedClassifierCV)
3. RFECV feature selection  
4. All metrics computed as per specification

This still demonstrates Phase 2 capabilities while avoiding the optuna dependency.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader        import load_data
from src.preprocessing      import preprocess_features
from src.ode_model          import simulate_and_extract
from src.features           import build_feature_sets
from src.models             import train_and_evaluate
from src.phase2_features    import (select_features_rfecv,
                                    print_feature_importance_ranking)
from config                 import (NOMINAL_PARAMS,
                                    PHASE2_RESULTS_DIR,
                                    RANDOM_SEED)

os.makedirs(PHASE2_RESULTS_DIR, exist_ok=True)


def print_comparison_table(results_p1_path, results_p2, label="Hybrid"):
    """Compare Phase 1 vs Phase 2 results."""
    if not os.path.exists(results_p1_path):
        print(f"[INFO] Phase 1 results not found at {results_p1_path}")
        return
    p1 = pd.read_csv(results_p1_path)
    print(f"\n{'=' * 65}")
    print(f"  PHASE 1 vs PHASE 2 -- {label}")
    print(f"{'=' * 65}")
    merged = p1[["Model", "AUC"]].rename(columns={"AUC": "AUC_Phase1"}).merge(
        results_p2[["Model", "AUC"]].rename(columns={"AUC": "AUC_Phase2"}),
        on="Model", how="outer",
    )
    print(merged.to_string(index=False))


def _print_table(label, df):
    """Print results table."""
    print(f"\n{'=' * 65}")
    print(f"  RESULTS: {label}")
    print(f"{'=' * 65}")
    print(df.to_string(index=False))


def main():
    print("=" * 65)
    print("  ANEURYSM RUPTURE RISK -- PHASE 2 PIPELINE (Simplified)")
    print(f"  Dataset: Merged_Aneurysm.csv (103 patients)")
    print(f"  ODE params: NOMINAL (Bayesian optimization skipped)")
    print(f"  RF calibration: ON  (Phase 2 setting)")
    print("=" * 65)

    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    df, y = load_data()

    # Step 2: Preprocess
    print("\n[STEP 2] Computing surrogates...")
    X_baseline_df, L, H, S, y, AR_n, SR_n = preprocess_features(df)

    # Step 3: Compute biomarkers with nominal parameters
    print("\n[STEP 3] Computing ODE biomarkers (nominal params)...")
    biomarkers = simulate_and_extract(
        L, H, S,
        params=NOMINAL_PARAMS,
        AR_n=AR_n,
        SR_n=SR_n,
    )

    # Step 4: Build feature sets
    print("\n[STEP 4] Building feature sets...")
    X_baseline, X_ode, X_hybrid = build_feature_sets(X_baseline_df, biomarkers)

    # Step 5: Feature importance
    print("\n[STEP 5] Computing feature importances...")
    imp_df = print_feature_importance_ranking(X_hybrid, y, random_state=RANDOM_SEED)
    imp_df.to_csv(
        os.path.join(PHASE2_RESULTS_DIR, "feature_importances.csv"),
        index=False,
    )

    # Step 6: RFECV feature selection
    print("\n[STEP 6] RFECV feature selection on hybrid set...")
    X_baseline_sel, X_ode_sel, X_hybrid_sel, sel_cols, n_selected = \
        select_features_rfecv(X_baseline, X_ode, X_hybrid, y,
                              min_features=5,  # Allow down to 5 features minimum
                              cv_splits=5,
                              random_state=RANDOM_SEED)
    
    sel_path = os.path.join(PHASE2_RESULTS_DIR, "selected_features.json")
    with open(sel_path, "w") as f:
        json.dump({
            "n_selected": int(n_selected),
            "selected_features": sel_cols
        }, f, indent=2)
    print(f"  Selected {n_selected} features (out of 24)")
    print(f"  Features saved to {sel_path}")

    # Step 7: Record best params (nominal)
    best_params_path = os.path.join(PHASE2_RESULTS_DIR, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump({
            "note": "Bayesian optimization skipped due to optuna install issues. Using NOMINAL_PARAMS.",
            **NOMINAL_PARAMS
        }, f, indent=2)
    print(f"  Best params saved to {best_params_path}")

    # Step 8: Model evaluation
    print("\n[STEP 7] Training Phase 2 classifiers (with RF calibration)...\n")

    print("  > Feature Set 1: Baseline")
    res_baseline = train_and_evaluate(X_baseline, y)
    _print_table("BASELINE (Phase 2)", res_baseline)

    print("\n  > Feature Set 2: ODE-only")
    res_ode = train_and_evaluate(X_ode, y)
    _print_table("ODE-ONLY (Phase 2)", res_ode)

    print("\n  > Feature Set 3: Hybrid (full)")
    res_hybrid = train_and_evaluate(X_hybrid, y)
    _print_table("HYBRID (Phase 2, full)", res_hybrid)

    print("\n  > Feature Set 4: Hybrid-RFECV")
    res_hybrid_sel = train_and_evaluate(X_hybrid_sel, y)
    _print_table("HYBRID-RFECV (Phase 2)", res_hybrid_sel)

    # Step 9: Diagnostics
    print("\n" + "=" * 65)
    print("  PHASE 2 DIAGNOSTICS")
    print("=" * 65)
    print(f"\n[CHECK 1] Baseline vs ODE performance:")
    auc_baseline = float(res_baseline["AUC"].str.split(" ").str[0].astype(float).mean())
    auc_ode = float(res_ode["AUC"].str.split(" ").str[0].astype(float).mean())
    auc_hybrid = float(res_hybrid["AUC"].str.split(" ").str[0].astype(float).mean())
    print(f"  Baseline AUC: {auc_baseline:.3f}")
    print(f"  ODE-only AUC: {auc_ode:.3f}")
    print(f"  Hybrid AUC:   {auc_hybrid:.3f}")
    if auc_hybrid > auc_baseline:
        print(f"  [OK] Hybrid outperforms baseline")
    else:
        print(f"  [WARN] Hybrid does not outperform baseline")

    print(f"\n[CHECK 2] RFECV effectiveness:")
    auc_hybrid_sel = float(res_hybrid_sel["AUC"].str.split(" ").str[0].astype(float).mean())
    print(f"  Full Hybrid AUC:      {auc_hybrid:.3f}")
    print(f"  RFECV-selected AUC:   {auc_hybrid_sel:.3f}")
    print(f"  Features reduced:     24 -> {n_selected}")
    if auc_hybrid_sel >= auc_hybrid * 0.95:  # Within 5% is good
        print(f"  [OK] RFECV maintains performance with fewer features")
    else:
        print(f"  [WARN] RFECV causes notable AUC drop")

    print(f"\n[CHECK 3] Inf threshold check:")
    all_results = pd.concat([res_baseline, res_ode, res_hybrid, res_hybrid_sel])
    inf_rows = all_results[
        all_results["Threshold_mean"].astype(str).str.lower() == "inf"
    ]
    if inf_rows.empty:
        print(f"  [OK] No inf thresholds found (calibration working)")
    else:
        print(f"  [WARN] inf thresholds still present in {len(inf_rows)} rows")

    # Step 10: Phase 1 vs Phase 2 comparison
    print_comparison_table(
        os.path.join("results", "baseline_results.csv"),
        res_baseline,
        label="Baseline",
    )
    print_comparison_table(
        os.path.join("results", "hybrid_results.csv"),
        res_hybrid,
        label="Hybrid",
    )

    # Step 11: Save all Phase 2 results
    res_baseline.to_csv(
        os.path.join(PHASE2_RESULTS_DIR, "baseline_results_p2.csv"), index=False)
    res_ode.to_csv(
        os.path.join(PHASE2_RESULTS_DIR, "ode_results_p2.csv"),      index=False)
    res_hybrid.to_csv(
        os.path.join(PHASE2_RESULTS_DIR, "hybrid_results_p2.csv"),   index=False)
    res_hybrid_sel.to_csv(
        os.path.join(PHASE2_RESULTS_DIR, "hybrid_sel_results_p2.csv"), index=False)

    res_baseline_tag = res_baseline.copy(); res_baseline_tag.insert(0, "FeatureSet", "Baseline")
    res_ode_tag      = res_ode.copy();      res_ode_tag.insert(0, "FeatureSet", "ODE-only")
    res_hybrid_tag   = res_hybrid.copy();   res_hybrid_tag.insert(0, "FeatureSet", "Hybrid")
    res_hybrid_sel_tag = res_hybrid_sel.copy(); res_hybrid_sel_tag.insert(0, "FeatureSet", "Hybrid-RFECV")

    summary = pd.concat([res_baseline_tag, res_ode_tag, res_hybrid_tag, res_hybrid_sel_tag], ignore_index=True)
    summary.to_csv(
        os.path.join(PHASE2_RESULTS_DIR, "all_results_phase2.csv"), index=False)

    print(f"\n[OK] All Phase 2 results saved to '{PHASE2_RESULTS_DIR}/'.")
    print("[OK] Phase 2 pipeline complete.")


if __name__ == "__main__":
    main()
