import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from config import NOMINAL_PARAMS
 
 
def aneurysm_ode(t, state, L_val, H_val, S_val, params):
    """Reduced-order ODE system (Paper Eq. 5-7)."""
    r, c, i = state
    dr_dt = params["a1"] * (S_val - 0.5) + params["a2"] * i  - params["a3"] * c
    dc_dt = params["b1"] * (1 - S_val)   - params["b2"] * i * c - params["b3"] * c
    di_dt = params["c1"] * L_val         + params["c2"] * H_val - params["c3"] * i
    return [dr_dt, dc_dt, di_dt]
 
 
def simulate_and_extract(L, H, S, params=None):
    """Simulate ODE for all patients and extract biomarkers (Paper Section III).
 
    Returns a dict with 10 biomarker arrays, each of length n_patients:
        r0, r_end, delta_r, i_max, AUC_i, c_min, c_end,
        I      (continuous mean excess ratio)   ← Bug 3 fix
        I_dur  (fraction of time i > c)         ← Bug 3 optimisation extra
        I_auc  (trapz area of excess curve)     ← Bug 3 optimisation extra
    """
    if params is None:
        params = NOMINAL_PARAMS
 
    n_patients = len(L)
 
    # ── BUG 2 OPT: data-derived S_REF (computed once, outside loop) ────────
    S_REF = np.percentile(S, 10)
    print(f"[INFO] S_REF (10th percentile of S) = {S_REF:.4f}")
    # ────────────────────────────────────────────────────────────────────────
 
    biomarkers = {
        "r0":    [], "r_end":  [], "delta_r": [],
        "i_max": [], "AUC_i":  [],
        "c_min": [], "c_end":  [],
        "I":     [],                # Bug 3 fix  — continuous mean excess ratio
        "I_dur": [],                # Bug 3 opt  — fraction of time i > c
        "I_auc": [],                # Bug 3 opt  — area under excess curve
    }
 
    t_eval = np.linspace(0, 1, 100)
 
    for idx in range(n_patients):
        # ── BUG 2 FIX + OPT: meaningful r₀ blend ──────────────────────────
        # Was: r0 = 0.3*S[idx] + 0.7*S[idx]  (always == S[idx])
        # Fix: r0 = 0.3*S_REF + 0.7*S[idx]   (Paper Eq. 8-10 intent)
        r0 = 0.3 * S_REF + 0.7 * S[idx]
        # ────────────────────────────────────────────────────────────────────
 
        c0 = 1.0
        i0 = 0.5 * (L[idx] + H[idx])
        y0 = [r0, c0, i0]
 
        sol = solve_ivp(
            fun=lambda t, y: aneurysm_ode(t, y, L[idx], H[idx], S[idx], params),
            t_span=[0, 1], y0=y0, method="RK45",
            dense_output=True, max_step=0.01
        )
 
        r_t, c_t, i_t = sol.sol(t_eval)
 
        # ── BUG 3 FIX + OPT: continuous I biomarkers ──────────────────────
        # excess(t) = max(0, i(t) - c(t))
        excess = np.maximum(0.0, i_t - c_t)
 
        # Fix  — mean excess ratio (continuous, replaces binary flag)
        I_mean_ratio = float(np.mean(excess / (c_t + 1e-8)))
 
        # Opt A — fraction of time-steps where inflammation exceeds wall integrity
        I_dur = float(np.mean(i_t > c_t))
 
        # Opt B — area under excess curve (total inflammatory burden over time)
        I_auc = float(trapezoid(excess, t_eval))
        # ────────────────────────────────────────────────────────────────────
 
        # Standard biomarkers
        biomarkers["r0"].append(float(r_t[0]))
        biomarkers["r_end"].append(float(r_t[-1]))
        biomarkers["delta_r"].append(float(r_t[-1] - r_t[0]))
        biomarkers["i_max"].append(float(np.max(i_t)))
        biomarkers["AUC_i"].append(float(trapezoid(i_t, t_eval)))
        biomarkers["c_min"].append(float(np.min(c_t)))
        biomarkers["c_end"].append(float(c_t[-1]))
 
        # I biomarkers (fixed + new)
        biomarkers["I"].append(I_mean_ratio)
        biomarkers["I_dur"].append(I_dur)
        biomarkers["I_auc"].append(I_auc)
 
    print("[OK] ODE simulation & biomarker extraction complete "
          f"(10 biomarkers, {n_patients} patients).")
    return biomarkers