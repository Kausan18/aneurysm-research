"""
main.py
Orchestrates the Phase 1 baseline pipeline for Merged_Aneurysm.csv.

Pipeline:
    data_loader -> preprocessing -> ode_model -> features -> models

CHANGES WITH WSS INTEGRATION:
  1. Data loading now joins Merged_Aneurysm.csv with Aneurysm_WSS_values_clean.csv
     on case_id and renames WSS_mean_dyn_cm2 to WSS_mean for use in pipeline.
  2. Baseline feature set now includes WSS_mean as 14th feature (was 13 before).
  3. H surrogate is now MinMaxNorm(WSS_mean) instead of MinMaxNorm(maxCurvature × tortuosity).
  4. Hybrid feature set is now 24 features (14 baseline + 10 ODE) instead of 23.

PREVIOUS FIXES:
  1. preprocess_features() returns AR_n and SR_n for use in ODE r0 calculation.
  2. simulate_and_extract() is called with AR_n=AR_n, SR_n=SR_n so that
     r0 = 0.3*SR_n + 0.7*AR_n per Paper Eq. 8.
  3. USE_CALIBRATION=False in models.py (Phase 2 feature, not Phase 1).
  4. ruptureStatus encoding fixed in data_loader.py (always map R/U strings).
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader   import load_data
from src.preprocessing import preprocess_features
from src.ode_model     import simulate_and_extract
from src.features      import build_feature_sets
from src.models        import train_and_evaluate
from config            import NOMINAL_PARAMS, RESULTS_DIR


def run_diagnostics(X_baseline, X_ode, X_hybrid, L, H, S, AR_n, SR_n):
    """Diagnostic checklist -- must pass before trusting any AUC numbers."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC CHECKLIST")
    print("=" * 60)

    print(f"\n[CHECK 1] Feature set shapes:")
    print(f"  Baseline : {X_baseline.shape}  (expected ~103 x 14)")
    print(f"  ODE-only : {X_ode.shape}        (expected  103 x 10)")
    print(f"  Hybrid   : {X_hybrid.shape}  (expected ~103 x 24)")
    assert X_baseline.shape[1] != X_ode.shape[1], \
        "FAIL: Baseline == ODE shape -- concat is broken!"
    assert X_hybrid.shape[1] == X_baseline.shape[1] + X_ode.shape[1], \
        "FAIL: Hybrid shape != Baseline + ODE!"
    print("  [OK] All shapes correct and distinct.")

    print(f"\n[CHECK 2] Surrogate std (must be > 0.05):")
    print(f"  S std: {S.std():.4f}  {'OK' if S.std() > 0.05 else '<< LOW'}")
    print(f"  L std: {L.std():.4f}  {'OK' if L.std() > 0.05 else '<< LOW'}")
    print(f"  H std: {H.std():.4f}  {'OK' if H.std() > 0.05 else '<< LOW'}")

    print(f"\n[CHECK 3] r0 inputs (Paper Eq. 8):")
    r0_vals = 0.3 * SR_n + 0.7 * AR_n
    print(f"  AR_n std  : {AR_n.std():.4f}")
    print(f"  SR_n std  : {SR_n.std():.4f}")
    print(f"  r0   std  : {r0_vals.std():.4f}  (must be > 0 -- not a constant)")
    print(f"  r0   range: [{r0_vals.min():.4f}, {r0_vals.max():.4f}]")
    assert r0_vals.std() > 0.01, "FAIL: r0 is near-constant -- check AR_n / SR_n!"
    print("  [OK] r0 has meaningful inter-patient variation.")

    print(f"\n[CHECK 4] ODE biomarker std values (flag if < 0.01):")
    bio_stds = X_ode.std().sort_values()
    for col, std_val in bio_stds.items():
        flag = "  ** LOW (degenerate) **" if std_val < 0.01 else ""
        print(f"  {col}: {std_val:.4f}{flag}")

    print("\n" + "=" * 60 + "\n")


def print_results_table(label: str, df: pd.DataFrame):
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {label}")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))


def main():
    print("=" * 60)
    print("  ANEURYSM RUPTURE RISK -- PHASE 1 BASELINE PIPELINE")
    print("  Dataset: Merged_Aneurysm.csv (103 patients)")
    print("  r0: 0.3*SR_n + 0.7*AR_n  (Paper Eq. 8)")
    print("  RF calibration: OFF  (Phase 1 setting)")
    print("=" * 60)

    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    df, y = load_data()

    # Step 2: Preprocess -- returns AR_n, SR_n in addition to previous outputs
    print("\n[STEP 2] Preprocessing & computing surrogates...")
    X_baseline_df, L, H, S, y, AR_n, SR_n = preprocess_features(df)

    # Step 3: Simulate ODE with correct Eq. 8 initial conditions
    print("\n[STEP 3] Simulating ODE system & extracting biomarkers...")
    biomarkers = simulate_and_extract(
        L, H, S,
        params=NOMINAL_PARAMS,
        AR_n=AR_n,
        SR_n=SR_n,
    )

    # Step 4: Build feature sets
    print("\n[STEP 4] Building feature sets...")
    X_baseline, X_ode, X_hybrid = build_feature_sets(X_baseline_df, biomarkers)

    # Step 5: Diagnostics
    run_diagnostics(X_baseline, X_ode, X_hybrid, L, H, S, AR_n, SR_n)

    # Step 6: Train & evaluate
    print("[STEP 5] Training classifiers (RepeatedStratifiedKFold 5x3)...\n")

    print("  > Feature Set 1: Baseline (geometry, 13 features)")
    res_baseline = train_and_evaluate(X_baseline, y)
    print_results_table("BASELINE (Set 1)", res_baseline)

    print("\n  > Feature Set 2: ODE-only (10 biomarkers)")
    res_ode = train_and_evaluate(X_ode, y)
    print_results_table("ODE-ONLY (Set 2)", res_ode)

    print("\n  > Feature Set 3: Hybrid (baseline + ODE, ~23 features)")
    res_hybrid = train_and_evaluate(X_hybrid, y)
    print_results_table("HYBRID (Set 3)", res_hybrid)

    # Step 7: Sanity check ODE != Hybrid
    ode_auc    = float(res_ode["AUC"].str.split(" ").str[0].astype(float).mean())
    hybrid_auc = float(res_hybrid["AUC"].str.split(" ").str[0].astype(float).mean())
    if abs(ode_auc - hybrid_auc) < 1e-6:
        print("[WARN] ODE and Hybrid AUCs are IDENTICAL -- concat may be broken!")
    else:
        print(f"\n[CHECK] ODE mean AUC={ode_auc:.3f} vs "
              f"Hybrid mean AUC={hybrid_auc:.3f} -- sets are distinct. [OK]")

    # Step 8: Save results
    res_baseline.to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False)
    res_ode.to_csv(      os.path.join(RESULTS_DIR, "ode_results.csv"),      index=False)
    res_hybrid.to_csv(   os.path.join(RESULTS_DIR, "hybrid_results.csv"),   index=False)

    res_baseline.insert(0, "FeatureSet", "Baseline")
    res_ode.insert(0, "FeatureSet", "ODE-only")
    res_hybrid.insert(0, "FeatureSet", "Hybrid")
    summary = pd.concat([res_baseline, res_ode, res_hybrid], ignore_index=True)
    summary.to_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index=False)

    print(f"\n[OK] All results saved to '{RESULTS_DIR}/'.")
    print("[OK] Phase 1 pipeline complete.")


if __name__ == "__main__":
    main()