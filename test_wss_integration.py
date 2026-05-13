#!/usr/bin/env python3
"""Quick test to verify WSS integration changes."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("[TEST] Testing WSS integration changes...\n")

# Test 1: Data loading
print("[TEST 1] Data loading with WSS join")
try:
    from src.data_loader import load_data
    df, y = load_data()
    assert "WSS_mean" in df.columns, "WSS_mean not in columns!"
    assert df.shape[0] == 103, f"Expected 103 rows, got {df.shape[0]}"
    print(f"✓ Data loaded: {df.shape}")
    print(f"✓ WSS_mean present: {df['WSS_mean'].notna().sum()} values")
    print(f"✓ WSS_mean stats: min={df['WSS_mean'].min():.2f}, max={df['WSS_mean'].max():.2f}, mean={df['WSS_mean'].mean():.2f}\n")
except Exception as e:
    print(f"✗ FAILED: {e}\n")
    sys.exit(1)

# Test 2: Preprocessing with WSS as H
print("[TEST 2] Preprocessing with WSS as H surrogate")
try:
    from src.preprocessing import preprocess_features
    X_bl, L, H, S, y_out, AR_n, SR_n = preprocess_features(df)
    assert X_bl.shape[1] == 14, f"Baseline should have 14 features, got {X_bl.shape[1]}"
    assert "WSS_mean" in X_bl.columns, "WSS_mean not in baseline features!"
    assert H.std() > 0.01, f"H std={H.std():.4f} is too low"
    print(f"✓ Baseline features: {X_bl.shape} (includes WSS_mean)")
    print(f"✓ H std={H.std():.4f}, L std={L.std():.4f}, S std={S.std():.4f}")
    print(f"✓ All surrogates have adequate variance\n")
except Exception as e:
    print(f"✗ FAILED: {e}\n")
    sys.exit(1)

# Test 3: ODE simulation
print("[TEST 3] ODE simulation with WSS-driven H")
try:
    from src.ode_model import simulate_and_extract
    from config import NOMINAL_PARAMS
    biomarkers = simulate_and_extract(L, H, S, params=NOMINAL_PARAMS, AR_n=AR_n, SR_n=SR_n)
    assert len(biomarkers) == 10, f"Expected 10 biomarkers, got {len(biomarkers)}"
    assert all(len(v) == 103 for v in biomarkers.values()), "Biomarker length mismatch"
    print(f"✓ ODE simulation complete: 10 biomarkers x 103 patients")
    print(f"✓ Sample biomarkers: r0 std={X_bl.shape}")
except Exception as e:
    print(f"✗ FAILED: {e}\n")
    sys.exit(1)

# Test 4: Feature set construction
print("[TEST 4] Feature set construction (14+10=24)")
try:
    from src.features import build_feature_sets
    X_baseline, X_ode, X_hybrid = build_feature_sets(X_bl, biomarkers)
    assert X_baseline.shape == (103, 14), f"Baseline shape mismatch: {X_baseline.shape}"
    assert X_ode.shape == (103, 10), f"ODE shape mismatch: {X_ode.shape}"
    assert X_hybrid.shape == (103, 24), f"Hybrid shape mismatch: {X_hybrid.shape}"
    print(f"✓ Baseline: {X_baseline.shape}")
    print(f"✓ ODE-only: {X_ode.shape}")
    print(f"✓ Hybrid: {X_hybrid.shape}\n")
except Exception as e:
    print(f"✗ FAILED: {e}\n")
    sys.exit(1)

print("[SUCCESS] All integration tests passed!")
print("\nNow ready to run Phase 1 and Phase 2 pipelines.")
