"""
ode_model.py
Paper mapping: Sections II.D, II.E, III
Defines ODE system, initial conditions, simulates trajectories, extracts 8 biomarkers.
"""

import numpy as np
from scipy.integrate import solve_ivp
from config import NOMINAL_PARAMS

def aneurysm_ode(t, state, L_val, H_val, S_val, params):
    """Reduced-order ODE system (Paper Eq. 5-7)."""
    r, c, i = state
    dr_dt = params["a1"] * (S_val - 0.5) + params["a2"] * i - params["a3"] * c
    dc_dt = params["b1"] * (1 - S_val) - params["b2"] * i * c - params["b3"] * c
    di_dt = params["c1"] * L_val + params["c2"] * H_val - params["c3"] * i
    return [dr_dt, dc_dt, di_dt]

def simulate_and_extract(L, H, S, params=None):
    """Simulate ODE for all patients and extract biomarkers (Paper Section III)."""
    if params is None:
        params = NOMINAL_PARAMS
        
    n_patients = len(L)
    biomarkers = {
        "r0": [], "r_end": [], "delta_r": [],
        "i_max": [], "AUC_i": [],
        "c_min": [], "c_end": [],
        "I": []
    }
    
    t_eval = np.linspace(0, 1, 100)
    
    for idx in range(n_patients):
        # Initial conditions (Paper Eq. 8-10)
        r0 = 0.3 * S[idx] + 0.7 * S[idx]  # Simplified: depends on S
        c0 = 1.0
        i0 = 0.5 * (L[idx] + H[idx])
        y0 = [r0, c0, i0]
        
        # Solve ODE
        sol = solve_ivp(
            fun=lambda t, y: aneurysm_ode(t, y, L[idx], H[idx], S[idx], params),
            t_span=[0, 1], y0=y0, method="RK45",
            dense_output=True, max_step=0.01
        )
        
        r_t, c_t, i_t = sol.sol(t_eval)
        
        # Extract biomarkers
        biomarkers["r0"].append(r_t[0])
        biomarkers["r_end"].append(r_t[-1])
        biomarkers["delta_r"].append(r_t[-1] - r_t[0])
        biomarkers["i_max"].append(np.max(i_t))
        biomarkers["AUC_i"].append(np.trapz(i_t, t_eval))
        biomarkers["c_min"].append(np.min(c_t))
        biomarkers["c_end"].append(c_t[-1])
        biomarkers["I"].append(1 if np.any(i_t > c_t) else 0)
        
    print("[OK] ODE simulation & biomarker extraction complete.")
    return biomarkers