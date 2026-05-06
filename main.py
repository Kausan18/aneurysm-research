"""
main.py
Orchestrates the baseline pipeline. Maps directly to Sections IV.A-IV.E.
"""

import os
import pandas as pd
import numpy as np
from src.data_loader import load_and_merge_data
from src.preprocessing import preprocess_features
from src.ode_model import simulate_and_extract, aneurysm_ode
from src.features import build_feature_sets
from src.models import train_and_evaluate
from src.visualization import plot_trajectories, plot_feature_importance
from config import NOMINAL_PARAMS, RESULTS_DIR

def main():
    print(" Starting Baseline Pipeline (Phase 1)")
    
    # 1. Load & merge data (Sec IV.A)
    df = load_and_merge_data()
    
    # 2. Preprocess & compute surrogates (Sec IV.A, II.C)
    X_norm, L, H, S, y, scaler = preprocess_features(df)
    
    # 3. Simulate ODE & extract biomarkers (Sec II.D-E, III)
    biomarkers = simulate_and_extract(L, H, S, params=NOMINAL_PARAMS)
    
    # 4. Build feature sets (Sec IV.C)
    X_baseline, X_ode, X_hybrid = build_feature_sets(X_norm, biomarkers)
    
    # 5. Train & evaluate (Sec IV.D)
    print("\n[INFO] Training Classifiers...")
    res_baseline = train_and_evaluate(X_baseline, y)
    res_ode = train_and_evaluate(X_ode, y)
    res_hybrid = train_and_evaluate(X_hybrid, y)
    
    # Save results
    res_baseline.to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False)
    res_ode.to_csv(os.path.join(RESULTS_DIR, "ode_results.csv"), index=False)
    res_hybrid.to_csv(os.path.join(RESULTS_DIR, "hybrid_results.csv"), index=False)
    
    # 6. Visualizations (Sec VI.B, VI.E)
    t_eval = np.linspace(0, 1, 100)
    # Pick one ruptured & one unruptured for plotting
    rupt_idx = np.where(y == 1)[0][0]
    unrupt_idx = np.where(y == 0)[0][0]
    
    # Re-simulate just for visualization
    from scipy.integrate import solve_ivp
    sol_r = solve_ivp(lambda t, s: aneurysm_ode(t, s, L[rupt_idx], H[rupt_idx], S[rupt_idx], NOMINAL_PARAMS), [0,1], [0.3*S[rupt_idx]+0.7*S[rupt_idx], 1.0, 0.5*(L[rupt_idx]+H[rupt_idx])], dense_output=True)
    sol_u = solve_ivp(lambda t, s: aneurysm_ode(t, s, L[unrupt_idx], H[unrupt_idx], S[unrupt_idx], NOMINAL_PARAMS), [0,1], [0.3*S[unrupt_idx]+0.7*S[unrupt_idx], 1.0, 0.5*(L[unrupt_idx]+H[unrupt_idx])], dense_output=True)
    
    plot_trajectories(sol_r, sol_u, t_eval)
    
    # Train final RF for feature importance
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_hybrid, y)
    plot_feature_importance(rf, X_hybrid.columns)
    
    print("\n[OK] Baseline pipeline complete! Check the `results/` folder.")

if __name__ == "__main__":
    main()