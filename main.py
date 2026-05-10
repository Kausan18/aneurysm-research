"""
main.py
Orchestrates the Phase 1 baseline pipeline for Merged_Aneurysm.csv.
Maps to Plan of Action Sections 1–7.

Pipeline:
    data_loader  →  preprocessing  →  ode_model  →  features  →  models
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader   import load_data
from src.preprocessing import preprocess_features
from src.ode_model     import simulate_and_extract
from src.features      import build_feature_sets
from src.models        import train_and_evaluate
from config            import NOMINAL_PARAMS, RESULTS_DIR


def run_diagnostics(X_baseline, X_ode, X_hybrid, L, H, S):
    """Diagnostic checklist — must pass before trusting any AUC numbers."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC CHECKLIST")
    print("=" * 60)

    # 1. Feature set shapes must all be different
    print(f"\n[CHECK 1] Feature set shapes:")
    print(f"  Baseline : {X_baseline.shape}  (expected ~103 × 13)")
    print(f"  ODE-only : {X_ode.shape}        (expected  103 × 10)")
    print(f"  Hybrid   : {X_hybrid.shape}  (expected ~103 × 23)")
    assert X_baseline.shape[1] != X_ode.shape[1], \
        "FAIL: Baseline == ODE shape -- concat is broken!"
    assert X_hybrid.shape[1] == X_baseline.shape[1] + X_ode.shape[1], \
        "FAIL: Hybrid shape != Baseline + ODE!"
    print("  [OK] All shapes correct and distinct.")

    # 2. Surrogate variance
    print(f"\n[CHECK 2] Surrogate std (must be > 0.05):")
    print(f"  S std: {S.std():.4f}  {'OK' if S.std() > 0.05 else '<< LOW'}")
    print(f"  L std: {L.std():.4f}  {'OK' if L.std() > 0.05 else '<< LOW'}")
    print(f"  H std: {H.std():.4f}  {'OK' if H.std() > 0.05 else '<< LOW'}")

    # 3. Biomarker variance
    print(f"\n[CHECK 3] ODE biomarker std values (flag if < 0.01):")
    bio_stds = X_ode.std().sort_values()
    for col, std_val in bio_stds.items():
        flag = "  ** LOW (degenerate) **" if std_val < 0.01 else ""
        print(f"  {col}: {std_val:.4f}{flag}")

    print("\n" + "=" * 60 + "\n")


def print_results_table(label: str, df: pd.DataFrame):
    """Pretty-print a results DataFrame."""
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {label}")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))


def main():
    print("=" * 60)
    print("  ANEURYSM RUPTURE RISK — PHASE 1 BASELINE PIPELINE")
    print("  Dataset: Merged_Aneurysm.csv (103 patients)")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[STEP 1] Loading data...")
    df, y = load_data()

    # ── 2. Preprocess & compute surrogates ────────────────────────────────────
    print("\n[STEP 2] Preprocessing & computing surrogates...")
    X_baseline_df, L, H, S, y = preprocess_features(df)

    # ── 3. Simulate ODE & extract biomarkers ──────────────────────────────────
    print("\n[STEP 3] Simulating ODE system & extracting biomarkers...")
    biomarkers = simulate_and_extract(L, H, S, params=NOMINAL_PARAMS)

    # ── 4. Build feature sets ─────────────────────────────────────────────────
    print("\n[STEP 4] Building feature sets...")
    X_baseline, X_ode, X_hybrid = build_feature_sets(X_baseline_df, biomarkers)

    # ── 5. Diagnostics ────────────────────────────────────────────────────────
    run_diagnostics(X_baseline, X_ode, X_hybrid, L, H, S)

    # ── 6. Train & evaluate ───────────────────────────────────────────────────
    print("[STEP 5] Training classifiers (RepeatedStratifiedKFold 5×3)...\n")

    print("  > Feature Set 1: Baseline (geometry, 13 features)")
    res_baseline = train_and_evaluate(X_baseline, y)
    print_results_table("BASELINE (Set 1)", res_baseline)

    print("\n  > Feature Set 2: ODE-only (10 biomarkers)")
    res_ode = train_and_evaluate(X_ode, y)
    print_results_table("ODE-ONLY (Set 2)", res_ode)

    print("\n  > Feature Set 3: Hybrid (baseline + ODE, ~23 features)")
    res_hybrid = train_and_evaluate(X_hybrid, y)
    print_results_table("HYBRID (Set 3)", res_hybrid)

    # ── 7. Check ODE ≠ Hybrid AUC (guards against concat bug) ────────────────
    ode_auc   = float(res_ode["AUC"].str.split(" ±").str[0].astype(float).mean())
    hybrid_auc = float(res_hybrid["AUC"].str.split(" ±").str[0].astype(float).mean())
    if abs(ode_auc - hybrid_auc) < 1e-6:
        print("[WARN] ODE and Hybrid AUCs are IDENTICAL - "
              "features.py concat may still be broken!")
    else:
        print(f"\n[CHECK] ODE mean AUC={ode_auc:.3f} vs "
              f"Hybrid mean AUC={hybrid_auc:.3f} - sets are distinct. [OK]")

    # ── 8. Save results ───────────────────────────────────────────────────────
    res_baseline.to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False)
    res_ode.to_csv(      os.path.join(RESULTS_DIR, "ode_results.csv"),      index=False)
    res_hybrid.to_csv(   os.path.join(RESULTS_DIR, "hybrid_results.csv"),   index=False)

    # Combined summary
    res_baseline.insert(0, "FeatureSet", "Baseline")
    res_ode.insert(0, "FeatureSet", "ODE-only")
    res_hybrid.insert(0, "FeatureSet", "Hybrid")
    summary = pd.concat([res_baseline, res_ode, res_hybrid], ignore_index=True)
    summary.to_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index=False)

    print(f"\n[OK] All results saved to '{RESULTS_DIR}/'.")
    print("     Files: baseline_results.csv, ode_results.csv, "
          "hybrid_results.csv, all_results.csv")
    print("\n[OK] Phase 1 pipeline complete.")


if __name__ == "__main__":
    main()