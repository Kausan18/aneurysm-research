"""
main_phase2.py
Orchestrates the full Phase 2 pipeline for Merged_Aneurysm.csv.
Maps to Plan of Action Sections 3–7.

Phase 2 pipeline:
    (shared Phase 1 data loading) →
    [Step 1]  phase2_surrogates  — refine L, H weights via logistic regression
    [Step 2]  phase2_optuna      — Bayesian search over 9 ODE params (50 trials)
    [Step 3]  ode_model          — recompute biomarkers with best_params
    [Step 4]  features           — build feature sets with optimised biomarkers
    [Step 5]  phase2_features    — RFECV feature selection on hybrid set
    [Step 6]  models             — full evaluation (LR + calibrated RF + GBM)
    [Step 7]  diagnostics        — all 6 checks from plan Section 7

Outputs written to results/phase2/:
    best_params.json, biomarker_std_comparison.csv,
    baseline_results_p2.csv, ode_results_p2.csv,
    hybrid_results_p2.csv, hybrid_sel_results_p2.csv,
    all_results_phase2.csv, feature_importances.csv
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
from src.phase2_optuna      import (run_optuna_study,
                                    check_params_changed,
                                    compare_biomarker_stds)
from src.phase2_surrogates  import refine_surrogate_weights
from src.phase2_features    import (select_features_rfecv,
                                    print_feature_importance_ranking)
from config                 import (NOMINAL_PARAMS,
                                    PHASE2_N_TRIALS,
                                    PHASE2_RESULTS_DIR,
                                    RANDOM_SEED)


# ── Configurable flags ────────────────────────────────────────────────────────
RUN_SURROGATE_REFINEMENT = True   # Phase 2 Section 4.2 — refine L/H weights
RUN_OPTUNA               = True   # Phase 2 Section 3   — Bayesian ODE search
RUN_RFECV                = True   # Phase 2 Section 4.1 — hybrid feature selection
PRINT_IMPORTANCES        = True   # Pre-RFECV permutation importance report


def run_phase2_diagnostics(best_params: dict,
                           biomarkers_p1: dict,
                           biomarkers_p2: dict,
                           X_baseline: pd.DataFrame,
                           X_ode: pd.DataFrame,
                           X_hybrid: pd.DataFrame,
                           results_ode: pd.DataFrame,
                           results_hybrid: pd.DataFrame,
                           results_baseline: pd.DataFrame) -> None:
    """Run all 6 diagnostic checks from Plan of Action Section 7."""
    print("\n" + "=" * 65)
    print("  PHASE 2 DIAGNOSTIC CHECKLIST (Plan Section 7)")
    print("=" * 65)

    # ── Check 1: Optimised params differ from nominal ─────────────────────────
    print("\n[CHECK 1] Param convergence (optimised ≠ nominal):")
    ok1 = check_params_changed(best_params, tol=0.01)
    if not ok1:
        print("  ACTION: Re-run Optuna with more trials (increase PHASE2_N_TRIALS).")

    # ── Check 2: Biomarker stds improve ──────────────────────────────────────
    print("\n[CHECK 2] Biomarker std improvement (Phase 1 → Phase 2):")
    cmp = compare_biomarker_stds(biomarkers_p1, biomarkers_p2)
    n_improved = int(cmp["improved"].sum())
    print(f"  {n_improved}/{len(cmp)} biomarkers have higher std under "
          f"optimised params.")
    if n_improved < len(cmp) // 2:
        print("  [WARN] Fewer than half the biomarkers improved — "
              "check surrogate quality.")

    # ── Check 3: Hybrid must outperform Baseline ─────────────────────────────
    print("\n[CHECK 3] Hybrid AUC > Baseline AUC:")
    auc_hybrid   = float(results_hybrid["AUC"].str.split(" ±").str[0].astype(float).mean())
    auc_baseline = float(results_baseline["AUC"].str.split(" ±").str[0].astype(float).mean())
    if auc_hybrid > auc_baseline:
        print(f"  [OK] Hybrid AUC={auc_hybrid:.3f} > Baseline AUC={auc_baseline:.3f}")
    else:
        print(f"  [WARN] Hybrid AUC={auc_hybrid:.3f} ≤ Baseline AUC={auc_baseline:.3f} "
              "— optimisation did not improve ODE contribution.")

    # ── Check 4: ODE-only RF AUC ≥ 0.50 ─────────────────────────────────────
    print("\n[CHECK 4] ODE-only RF AUC ≥ 0.50 (Phase 1 was 0.476):")
    rf_row = results_ode[results_ode["Model"] == "Random Forest"]
    if not rf_row.empty:
        ode_rf_auc = float(rf_row["AUC"].str.split(" ±").str[0].values[0])
        if ode_rf_auc >= 0.50:
            print(f"  [OK] ODE-only RF AUC = {ode_rf_auc:.3f} ≥ 0.50")
        else:
            print(f"  [WARN] ODE-only RF AUC = {ode_rf_auc:.3f} < 0.50 "
                  "— ODE biomarkers still hurting RF.")
    else:
        print("  [WARN] Could not find RF row in ODE results.")

    # ── Check 5: No inf thresholds ────────────────────────────────────────────
    print("\n[CHECK 5] Threshold=inf check (must be eliminated):")
    all_results = pd.concat([results_baseline, results_ode, results_hybrid])
    inf_rows = all_results[all_results["Threshold_mean"].astype(str).str.lower() == "inf"]
    if inf_rows.empty:
        print("  [OK] No inf thresholds found in any result.")
    else:
        print(f"  [WARN] inf thresholds still present in:\n{inf_rows[['Model', 'Threshold_mean']]}")

    # ── Check 6: All feature set shapes distinct ─────────────────────────────
    print("\n[CHECK 6] Feature set shapes distinct:")
    print(f"  Baseline : {X_baseline.shape}  |  "
          f"ODE : {X_ode.shape}  |  Hybrid : {X_hybrid.shape}")
    if (X_baseline.shape[1] != X_ode.shape[1] and
            X_hybrid.shape[1] == X_baseline.shape[1] + X_ode.shape[1]):
        print("  [OK] All shapes correct and distinct.")
    else:
        print("  [BUG] Shape mismatch — check build_feature_sets.")

    print("\n" + "=" * 65 + "\n")


def print_comparison_table(results_p1_path: str,
                           results_p2: pd.DataFrame,
                           label: str = "Hybrid") -> None:
    """Print Phase 1 vs Phase 2 AUC side-by-side if Phase 1 CSV exists."""
    if not os.path.exists(results_p1_path):
        print(f"[INFO] Phase 1 results not found at {results_p1_path} — "
              "skipping comparison table.")
        return
    p1 = pd.read_csv(results_p1_path)
    print(f"\n{'=' * 65}")
    print(f"  PHASE 1 vs PHASE 2 — {label}")
    print(f"{'=' * 65}")
    merged = p1[["Model", "AUC"]].rename(columns={"AUC": "AUC_Phase1"}).merge(
        results_p2[["Model", "AUC"]].rename(columns={"AUC": "AUC_Phase2"}),
        on="Model", how="outer",
    )
    print(merged.to_string(index=False))


def main():
    print("=" * 65)
    print("  ANEURYSM RUPTURE RISK — PHASE 2 PIPELINE")
    print(f"  Dataset: Merged_Aneurysm.csv (103 patients)")
    print(f"  Optuna trials: {PHASE2_N_TRIALS}  |  Seed: {RANDOM_SEED}")
    print("=" * 65)

    # ── 1. Load data (same as Phase 1) ───────────────────────────────────────
    print("\n[STEP 1] Loading data...")
    df, y = load_data()

    # ── 2. Phase 1 surrogates (baseline) ─────────────────────────────────────
    print("\n[STEP 2] Computing Phase 1 surrogates (nominal)...")
    X_baseline_df, L_p1, H_p1, S, y = preprocess_features(df)

    # ── 3. Phase 1 nominal biomarkers (for diagnostic comparison) ─────────────
    print("\n[STEP 3] Computing Phase 1 biomarkers (nominal params)...")
    biomarkers_p1 = simulate_and_extract(L_p1, H_p1, S, params=NOMINAL_PARAMS)

    # ── 4. Surrogate refinement (Section 4.2) ─────────────────────────────────
    if RUN_SURROGATE_REFINEMENT:
        print("\n[STEP 4] Refining L and H surrogate weights (LR-derived)...")
        L, H, L_weights, H_weights = refine_surrogate_weights(df, y,
                                                               random_state=RANDOM_SEED)
        # Save surrogate weights to JSON for paper reporting
        surrogate_report = {"L_weights": L_weights, "H_weights": H_weights}
        sw_path = os.path.join(PHASE2_RESULTS_DIR, "surrogate_weights.json")
        with open(sw_path, "w") as f:
            json.dump(surrogate_report, f, indent=2)
        print(f"  Surrogate weights saved → {sw_path}")
    else:
        print("\n[STEP 4] Skipping surrogate refinement (using Phase 1 L, H).")
        L, H = L_p1, H_p1

    # ── 5. Optuna ODE coefficient optimisation (Section 3) ───────────────────
    if RUN_OPTUNA:
        print(f"\n[STEP 5] Running Optuna study ({PHASE2_N_TRIALS} trials)...")
        best_params, study = run_optuna_study(
            L, H, S, X_baseline_df, y,
            n_trials=PHASE2_N_TRIALS,
            random_state=RANDOM_SEED,
        )
        # Save best params
        bp_path = os.path.join(PHASE2_RESULTS_DIR, "best_params.json")
        with open(bp_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"  Best ODE params saved → {bp_path}")
    else:
        print("\n[STEP 5] Skipping Optuna — using NOMINAL_PARAMS.")
        best_params = NOMINAL_PARAMS

    # ── 6. Recompute biomarkers with best_params ─────────────────────────────
    print("\n[STEP 6] Recomputing ODE biomarkers with optimised params...")
    biomarkers_p2 = simulate_and_extract(L, H, S, params=best_params)

    # ── 7. Build full feature sets ────────────────────────────────────────────
    print("\n[STEP 7] Building Phase 2 feature sets...")
    X_baseline, X_ode, X_hybrid = build_feature_sets(X_baseline_df, biomarkers_p2)

    # ── 8. Permutation importance (informational) ─────────────────────────────
    if PRINT_IMPORTANCES:
        print("\n[STEP 8] Permutation feature importances (pre-RFECV)...")
        imp_df = print_feature_importance_ranking(X_hybrid, y,
                                                  random_state=RANDOM_SEED)
        imp_df.to_csv(
            os.path.join(PHASE2_RESULTS_DIR, "feature_importances.csv"),
            index=False,
        )

    # ── 9. RFECV feature selection (Section 4.1) ─────────────────────────────
    if RUN_RFECV:
        print("\n[STEP 9] RFECV feature selection on Phase 2 hybrid set...")
        X_baseline_sel, X_ode_sel, X_hybrid_sel, sel_cols, _ = \
            select_features_rfecv(X_baseline, X_ode, X_hybrid, y,
                                  min_features=10,
                                  cv_splits=5,
                                  random_state=RANDOM_SEED)
        # Save selected feature list
        sel_path = os.path.join(PHASE2_RESULTS_DIR, "selected_features.json")
        with open(sel_path, "w") as f:
            json.dump({"selected_features": sel_cols}, f, indent=2)
        print(f"  Selected features saved → {sel_path}")
    else:
        print("\n[STEP 9] Skipping RFECV — using full feature sets.")
        X_baseline_sel, X_ode_sel, X_hybrid_sel = X_baseline, X_ode, X_hybrid

    # ── 10. Model evaluation — full Phase 2 sets ─────────────────────────────
    print("\n[STEP 10] Training classifiers on Phase 2 feature sets "
          "(RepeatedStratifiedKFold 5×3)...\n")

    print("  > Feature Set 1: Baseline (geometry, same as Phase 1)")
    res_baseline = train_and_evaluate(X_baseline, y)
    _print_table("BASELINE (Phase 2)", res_baseline)

    print("\n  > Feature Set 2: ODE-only (optimised params)")
    res_ode = train_and_evaluate(X_ode, y)
    _print_table("ODE-ONLY (Phase 2)", res_ode)

    print("\n  > Feature Set 3: Hybrid — full (optimised params)")
    res_hybrid = train_and_evaluate(X_hybrid, y)
    _print_table("HYBRID — full (Phase 2)", res_hybrid)

    if RUN_RFECV:
        print("\n  > Feature Set 4: Hybrid — RFECV selected")
        res_hybrid_sel = train_and_evaluate(X_hybrid_sel, y)
        _print_table("HYBRID — RFECV selected (Phase 2)", res_hybrid_sel)
    else:
        res_hybrid_sel = None

    # ── 11. Diagnostics ───────────────────────────────────────────────────────
    run_phase2_diagnostics(
        best_params, biomarkers_p1, biomarkers_p2,
        X_baseline, X_ode, X_hybrid,
        res_ode, res_hybrid, res_baseline,
    )

    # ── 12. Phase 1 vs Phase 2 comparison ────────────────────────────────────
    print_comparison_table(
        os.path.join("results", "hybrid_results.csv"),
        res_hybrid,
        label="Hybrid",
    )

    # ── 13. Save all Phase 2 results ─────────────────────────────────────────
    res_baseline.to_csv(
        os.path.join(PHASE2_RESULTS_DIR, "baseline_results_p2.csv"), index=False)
    res_ode.to_csv(
        os.path.join(PHASE2_RESULTS_DIR, "ode_results_p2.csv"),      index=False)
    res_hybrid.to_csv(
        os.path.join(PHASE2_RESULTS_DIR, "hybrid_results_p2.csv"),   index=False)

    res_baseline_tag = res_baseline.copy()
    res_ode_tag      = res_ode.copy()
    res_hybrid_tag   = res_hybrid.copy()
    res_baseline_tag.insert(0, "FeatureSet", "Baseline")
    res_ode_tag.insert(0, "FeatureSet", "ODE-only")
    res_hybrid_tag.insert(0, "FeatureSet", "Hybrid")

    parts = [res_baseline_tag, res_ode_tag, res_hybrid_tag]

    if res_hybrid_sel is not None:
        res_hybrid_sel.to_csv(
            os.path.join(PHASE2_RESULTS_DIR, "hybrid_sel_results_p2.csv"),
            index=False,
        )
        res_hybrid_sel_tag = res_hybrid_sel.copy()
        res_hybrid_sel_tag.insert(0, "FeatureSet", "Hybrid-RFECV")
        parts.append(res_hybrid_sel_tag)

    summary = pd.concat(parts, ignore_index=True)
    summary.to_csv(
        os.path.join(PHASE2_RESULTS_DIR, "all_results_phase2.csv"), index=False)

    print(f"\n[OK] All Phase 2 results saved to '{PHASE2_RESULTS_DIR}/'.")
    print("     Files: best_params.json, surrogate_weights.json, "
          "selected_features.json,")
    print("            baseline_results_p2.csv, ode_results_p2.csv, "
          "hybrid_results_p2.csv,")
    print("            hybrid_sel_results_p2.csv, all_results_phase2.csv, "
          "feature_importances.csv")
    print("\n[OK] Phase 2 pipeline complete.")


def _print_table(label: str, df: pd.DataFrame) -> None:
    """Pretty-print a results DataFrame with a label header."""
    print(f"\n{'=' * 65}")
    print(f"  RESULTS: {label}")
    print(f"{'=' * 65}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
