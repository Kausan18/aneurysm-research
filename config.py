"""
config.py
Paper mapping: Sections II.E, II.F, IV.C, IV.D
Stores nominal parameters, file paths, column mappings, and reproducibility settings.

Updated for Merged_Aneurysm.csv (103 patients, 44R / 59U).
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join("data")
RESULTS_DIR = os.path.join("results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Nominal ODE Parameters (Paper Eq. 5-7) ───────────────────────────────────
# Phase 1: fixed nominal values from the paper.
# Phase 2: these will be Bayesian-optimised over param_space.
NOMINAL_PARAMS = {
    # dr/dt = a1*(S - 0.5) + a2*i - a3*c
    "a1": 0.8,  "a2": 0.9,  "a3": 0.7,
    # dc/dt = b1*(1 - S)   - b2*i*c - b3*c
    "b1": 0.6,  "b2": 0.8,  "b3": 0.2,
    # di/dt = c1*L          + c2*H   - c3*i
    "c1": 1.0,  "c2": 0.7,  "c3": 0.9,
}

# ── Surrogate Weights (data-derived from Merged_Aneurysm.csv) ─────────────────
# S = S_ar * AR_n + S_sr * SR_n
#   Derived via logistic regression on this cohort:
#   SR (r=+0.196, p=0.048) dominates; AR (r=+0.046, p=0.642) near-zero.
# L = MinMax(1 - WSS_n)
#   Low-shear damage: low WSS -> high damage risk; WSS from Aneurysm_WSS_values_clean.csv (in Pascal).
# H = MinMax(WSS_mean)
#   High-shear damage driver from Aneurysm_WSS_values_clean.csv (in Pascal: 1 dyn/cm^2 = 0.1 Pa).
SURROGATE_WEIGHTS = {
    "S_ar": 0.025,   # logistic-regression weight for AR_n
    "S_sr": 0.975,   # logistic-regression weight for SR_n
    # L and H are computed entirely in preprocessing.py (no scalar weights needed)
}

# ── Column Mapping for Merged_Aneurysm.csv ────────────────────────────────────
COLUMN_MAP = {
    "morph": {
        "AR":                   "aspectRatio_star",
        "SR":                   "sizeRatio_star",
        "ostiumShapeFactor":    "ostiumShapeFactor",
        "sacVolume":            "sacVolume",
        "ellipsoidMinSemiaxis": "ellipsoidMinSemiaxis",
    },
    "vessel": {
        "minRadius":    "minRadius",
        "vesselDiameter": "vesselDiameter",
        "meanRadius":   "meanRadius",
        "maxRadius":    "maxRadius",
        "tortuosity":   "tortuosity",
        "maxCurvature": "maxCurvature",
        "meanCurvature": "meanCurvature",
        "length":       "length",
    },
    "clinical": {
        "RUPTURE":  "ruptureStatus",   # encode: R=1, U=0
        "age":      "age",
        "sex":      "sex",
        "location": "aneurysmLocation",
        "type":     "aneurysmType",
    },
}

# ── Baseline Feature Columns (Set 1) ─────────────────────────────────────────
# These 13 geometry/morphological + hemodynamic columns form the baseline feature set.
# ellipsoidMinSemiaxis has 2 missing rows -> imputed with median in preprocessing.
# WSS_mean is now the hemodynamic driver, sourced from Aneurysm_WSS_values_clean.csv.
# (Replaced minRadius proxy with real mean WSS values.)
BASELINE_FEATURES = [
    "vesselDiameter",
    "meanRadius",
    "maxRadius",
    "tortuosity",
    "length",
    "maxCurvature",
    "meanCurvature",
    "sizeRatio_star",
    "aspectRatio_star",
    "ostiumShapeFactor",
    "ellipsoidMinSemiaxis",
    "sacVolume",
    "WSS_mean",
]

# -- CFD Columns to Drop (95/103 missing -- not usable) ----------------
CFD_DROP_COLS = [
    "sacMinPressure", "sacMaxPressure", "sacMeanPressure",
    "sacMaxSpeed",    "sacMeanSpeed",
    "sacMinTAWSS",    "sacMaxTAWSS",    "sacMeanTAWSS",
    "sacMinOSI",      "sacMaxOSI",      "sacMeanOSI",
    "minPressure",    "maxPressure",    "meanPressure",
    "maxSpeed",       "meanSpeed",
    "minTAWSS",       "maxTAWSS",       "meanTAWSS",
    "minOSI",         "maxOSI",         "meanOSI",
]

# ── ID / Duplicate Columns to Drop ───────────────────────────────────────────
ID_DROP_COLS = ["case_id", "patient_id", "vesselName"]

# ── Phase 2: Optuna Search Space ──────────────────────────────────────────────
# Each key maps to (low, high) inclusive bounds for TPE continuous search.
# All parameters [0.1, 2.0] except b3 [0.01, 1.0] per specification.
PHASE2_PARAM_BOUNDS = {
    "a1": (0.1, 2.0),   # Shape -> wall stress          (nominal 0.8)
    "a2": (0.1, 2.0),   # Inflammation -> wall stress   (nominal 0.9)
    "a3": (0.1, 2.0),   # Integrity suppresses stress   (nominal 0.7)
    "b1": (0.1, 2.0),   # Normal shape -> integrity      (nominal 0.6)
    "b2": (0.1, 2.0),   # Inflammation degrades integr. (nominal 0.8)
    "b3": (0.01, 1.0),  # Passive integrity decay       (nominal 0.2)
    "c1": (0.1, 2.0),   # Low shear -> inflammation      (nominal 1.0)
    "c2": (0.1, 2.0),   # Haemodynamic stress -> inflam. (nominal 0.7)
    "c3": (0.1, 2.0),   # Natural inflammation resolution(nominal 0.9)
}

# Number of Optuna trials (80 per plan; 50 minimum per previous versions)
PHASE2_N_TRIALS = 80

# Output directory for Phase 2 results (separate from Phase 1 results/)
PHASE2_RESULTS_DIR = os.path.join("results", "phase2")
os.makedirs(PHASE2_RESULTS_DIR, exist_ok=True)