
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
    