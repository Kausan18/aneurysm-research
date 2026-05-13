"""
ode_model.py
Paper mapping: Sections II.D-E, III

ODE system is UNCHANGED (Paper Eq. 5-7).

FIX: r0 now uses Paper Eq. 8 correctly:
    r0 = 0.3 * SR_n + 0.7 * AR_n
    where SR_n = MinMax-normalised sizeRatio_star
          AR_n = MinMax-normalised aspectRatio_star

Previous code had r0 = 0.3*S_REF + 0.7*S[idx] which blended the composite
surrogate S with itself (via a population anchor) -- not in the paper at all.
The paper uses SR_n and AR_n as *separate* inputs to r0, not the same S twice.

The S_REF population anchor is removed entirely.

simulate_and_extract() accepts AR_n and SR_n as explicit parameters so both
Phase 1 and Phase 2 callers can pass them in.

10 biomarkers extracted:
  r0, r_end, delta_r, i_max, AUC_i, c_min, c_end,
  I  (continuous mean excess ratio),
  I_dur (fraction of time i > c),
  I_auc (trapz area of excess curve)
"""

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from config import NOMINAL_PARAMS


def aneurysm_ode(t, state, L_val, H_val, S_val, params):
    """Reduced-order ODE system (Paper Eq. 5-7).

    dr/dt = a1*(S - 0.5) + a2*i - a3*c
    dc/dt = b1*(1 - S)   - b2*i*c - b3*c
    di/dt = c1*L          + c2*H  - c3*i
    """
    r, c, i = state
    dr_dt = params["a1"] * (S_val - 0.5) + params["a2"] * i  - params["a3"] * c
    dc_dt = params["b1"] * (1 - S_val)   - params["b2"] * i * c - params["b3"] * c
    di_dt = params["c1"] * L_val         + params["c2"] * H_val - params["c3"] * i
    return [dr_dt, dc_dt, di_dt]


def simulate_and_extract(L, H, S, params=None, AR_n=None, SR_n=None):
    """Simulate ODE for all patients and extract 10 biomarkers (Paper Section III).

    Parameters
    ----------
    L, H, S : np.ndarray (n_patients,)
        Geometry-derived surrogates in [0, 1].
    params  : dict or None
        ODE coefficient dict. Defaults to NOMINAL_PARAMS from config.
    AR_n    : np.ndarray (n_patients,)
        MinMax-normalised aspectRatio_star. Used for r0 per Paper Eq. 8.
        Pass from preprocess_features() return value.
    SR_n    : np.ndarray (n_patients,)
        MinMax-normalised sizeRatio_star. Used for r0 per Paper Eq. 8.
        Pass from preprocess_features() return value.

    Returns
    -------
    biomarkers : dict with 10 keys, each a list of length n_patients:
        r0, r_end, delta_r, i_max, AUC_i, c_min, c_end, I, I_dur, I_auc
    """
    if params is None:
        params = NOMINAL_PARAMS

    n_patients = len(L)

    # Validate AR_n / SR_n -- both are required for correct Eq. 8
    if AR_n is None or SR_n is None:
        raise ValueError(
            "[ode_model] AR_n and SR_n are required for r0 = 0.3*SR_n + 0.7*AR_n "
            "(Paper Eq. 8). Pass them from preprocess_features()."
        )

    print("[INFO] r0 = 0.3*SR_n + 0.7*AR_n  (Paper Eq. 8 -- correct)")

    biomarkers = {
        "r0":    [], "r_end":  [], "delta_r": [],
        "i_max": [], "AUC_i":  [],
        "c_min": [], "c_end":  [],
        "I":     [],   # continuous mean excess ratio
        "I_dur": [],   # fraction of time i > c
        "I_auc": [],   # area under excess curve
    }

    t_eval = np.linspace(0, 1, 100)

    for idx in range(n_patients):
        # Paper Eq. 8-10: initial conditions
        r0 = 0.3 * SR_n[idx] + 0.7 * AR_n[idx]   # Eq. 8 CORRECT
        c0 = 1.0                                    # Eq. 9
        i0 = 0.5 * (L[idx] + H[idx])              # Eq. 10
        y0 = [r0, c0, i0]

        sol = solve_ivp(
            fun=lambda t, y: aneurysm_ode(t, y, L[idx], H[idx], S[idx], params),
            t_span=[0, 1],
            y0=y0,
            method="RK45",
            dense_output=True,
            max_step=0.01,
        )

        r_t, c_t, i_t = sol.sol(t_eval)

        # Continuous I biomarkers
        excess       = np.maximum(0.0, i_t - c_t)
        I_mean_ratio = float(np.mean(excess / (c_t + 1e-8)))
        I_dur        = float(np.mean(i_t > c_t))
        I_auc        = float(trapezoid(excess, t_eval))

        biomarkers["r0"].append(float(r_t[0]))
        biomarkers["r_end"].append(float(r_t[-1]))
        biomarkers["delta_r"].append(float(r_t[-1] - r_t[0]))
        biomarkers["i_max"].append(float(np.max(i_t)))
        biomarkers["AUC_i"].append(float(trapezoid(i_t, t_eval)))
        biomarkers["c_min"].append(float(np.min(c_t)))
        biomarkers["c_end"].append(float(c_t[-1]))
        biomarkers["I"].append(I_mean_ratio)
        biomarkers["I_dur"].append(I_dur)
        biomarkers["I_auc"].append(I_auc)

    print(f"[OK] ODE simulation complete: 10 biomarkers x {n_patients} patients.")
    return biomarkers