"""
ode_model.py
Paper mapping: Sections II.D-E, III

ODE system is UNCHANGED (Paper Eq. 5-7).
Revisions applied:
  - Revised initial conditions: r0 = 0.3*S_REF + 0.7*S[idx]  (Bug 2 fix)
    where S_REF = 10th percentile of S (data-derived population anchor)
  - 10 biomarkers including continuous I variants (Bug 3 fix)
  - Accepts optional params dict for Phase 2 Bayesian optimisation
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


def simulate_and_extract(L, H, S, params=None):
    """Simulate ODE for all patients and extract 10 biomarkers (Paper Section III).

    Parameters
    ----------
    L, H, S : np.ndarray (n_patients,)
        Geometry-derived surrogates in [0, 1].
    params  : dict or None
        ODE coefficient dict. If None, uses NOMINAL_PARAMS from config.

    Returns
    -------
    biomarkers : dict
        10 keys, each a Python list of length n_patients:
          r0, r_end, delta_r, i_max, AUC_i, c_min, c_end,
          I (continuous mean excess ratio),
          I_dur (fraction of time i > c),
          I_auc (trapz area of excess curve)
    """
    if params is None:
        params = NOMINAL_PARAMS

    n_patients = len(L)

    # ── Data-derived population anchor (Bug 2 fix) ────────────────────────────
    # S_REF = 10th percentile of S across the cohort.
    # This anchors r0 to the population, making the blend r0 = 0.3*S_REF + 0.7*S[idx]
    # meaningful (previously was 0.3*S[idx] + 0.7*S[idx] = S[idx], a no-op).
    S_REF = float(np.percentile(S, 10))
    print(f"[INFO] S_REF (10th percentile of S) = {S_REF:.4f}")

    biomarkers = {
        "r0":    [], "r_end":  [], "delta_r": [],
        "i_max": [], "AUC_i":  [],
        "c_min": [], "c_end":  [],
        "I":     [],   # Bug 3 fix  — continuous mean excess ratio (replaces binary)
        "I_dur": [],   # Bug 3 opt  — fraction of time i > c
        "I_auc": [],   # Bug 3 opt  — area under excess curve
    }

    t_eval = np.linspace(0, 1, 100)

    for idx in range(n_patients):
        # ── Revised initial conditions (Bug 2 fix) ────────────────────────────
        r0 = 0.3 * S_REF + 0.7 * S[idx]   # meaningful population/individual blend
        c0 = 1.0                            # healthy wall integrity at t=0
        i0 = 0.5 * (L[idx] + H[idx])       # initial inflammation from surrogates
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

        # ── Continuous I biomarkers (Bug 3 fix) ───────────────────────────────
        # excess(t) = max(0, i(t) - c(t))
        excess = np.maximum(0.0, i_t - c_t)

        I_mean_ratio = float(np.mean(excess / (c_t + 1e-8)))   # mean excess ratio
        I_dur        = float(np.mean(i_t > c_t))               # fraction time i > c
        I_auc        = float(trapezoid(excess, t_eval))         # total excess burden

        # ── Standard biomarkers ───────────────────────────────────────────────
        biomarkers["r0"].append(float(r_t[0]))
        biomarkers["r_end"].append(float(r_t[-1]))
        biomarkers["delta_r"].append(float(r_t[-1] - r_t[0]))
        biomarkers["i_max"].append(float(np.max(i_t)))
        biomarkers["AUC_i"].append(float(trapezoid(i_t, t_eval)))
        biomarkers["c_min"].append(float(np.min(c_t)))
        biomarkers["c_end"].append(float(c_t[-1]))

        # ── I biomarkers ──────────────────────────────────────────────────────
        biomarkers["I"].append(I_mean_ratio)
        biomarkers["I_dur"].append(I_dur)
        biomarkers["I_auc"].append(I_auc)

    print(f"[OK] ODE simulation complete: 10 biomarkers × {n_patients} patients.")
    return biomarkers