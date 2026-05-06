"""
visualization.py
Paper mapping: Sections VI.B, VI.E
Plots ODE trajectories and feature importance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from config import RESULTS_DIR

def plot_trajectories(sol_rupt, sol_unrupt, t_eval, save_path="trajectories.png"):
    """Plot r(t), c(t), i(t) for one ruptured and one unruptured sample."""
    r_r, c_r, i_r = sol_rupt.sol(t_eval)
    r_u, c_u, i_u = sol_unrupt.sol(t_eval)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    axes[0,0].plot(t_eval, r_r, 'r-', lw=2); axes[0,0].set_title("Ruptured: r(t)")
    axes[0,1].plot(t_eval, r_u, 'g-', lw=2); axes[0,1].set_title("Unruptured: r(t)")
    axes[1,0].plot(t_eval, c_r, 'b-', lw=2); axes[1,0].plot(t_eval, i_r, 'r--', lw=2); axes[1,0].set_title("Ruptured: c(t) & i(t)")
    axes[1,1].plot(t_eval, c_u, 'b-', lw=2); axes[1,1].plot(t_eval, i_u, 'g--', lw=2); axes[1,1].set_title("Unruptured: c(t) & i(t)")
    axes[2,0].plot(t_eval, r_r, 'r-', lw=2, label="Ruptured"); axes[2,0].plot(t_eval, r_u, 'g-', lw=2, label="Unruptured"); axes[2,0].legend()
    axes[2,1].plot(t_eval, c_r, 'b-', lw=2); axes[2,1].plot(t_eval, i_r, 'r--', lw=2); axes[2,1].set_title("Instability Check (i>c?)")
    
    for ax in axes.flat: ax.grid(True, alpha=0.3); ax.set_xlabel("Normalized Time")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, save_path), dpi=300)
    print(f"[OK] Trajectory plot saved to {save_path}")

def plot_feature_importance(model, feature_names, save_path="feature_importance.png"):
    """Plot top features from Random Forest hybrid model."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature Importance (Hybrid Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, save_path), dpi=300)
    print(f"[OK] Feature importance plot saved to {save_path}")