import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    balanced_accuracy_score, roc_curve,
)
 
# ── SMOTE import (optional — graceful fallback if imbalanced-learn not installed)
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False
    print("[WARN] imbalanced-learn not installed — SMOTE skipped. "
          "Install with: pip install imbalanced-learn")
 
 
def _apply_smote(X_tr, y_tr, random_state=42):
    """Apply SMOTE only when the minority class has enough samples."""
    if not _SMOTE_AVAILABLE:
        return X_tr, y_tr
    minority_count = int(np.sum(y_tr == 1))
    # k_neighbors=3 but cap at minority_count - 1 to avoid ValueError
    k = min(3, minority_count - 1)
    if k < 1:
        return X_tr, y_tr
    sm = SMOTE(k_neighbors=k, random_state=random_state)
    return sm.fit_resample(X_tr, y_tr)
 
 
def _best_threshold_youden(y_true, y_prob):
    """Return threshold that maximises Youden's J = Sensitivity + Specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr                          # Youden's J
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])
 
 
def train_and_evaluate(X, y, cv_folds=5, random_state=42):
    """Train 3 classifiers with stratified CV and return metrics DataFrame.
 
    Metrics reported per model:
        AUC             — mean ± std across folds (default 0.5 threshold)
        Accuracy        — mean at default threshold
        F1              — mean at default threshold
        Balanced_Accuracy_default  — at 0.5 threshold (shows baseline)
        Balanced_Accuracy_tuned    — at per-fold Youden-optimal threshold
        Threshold_mean  — mean optimal threshold across folds
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
 
    # ── BUG 4 FIX: class_weight / sample_weight on all three classifiers ──
    classifiers = {
        "Logistic Regression": LogisticRegression(
            penalty="l2", solver="lbfgs",
            max_iter=1000, random_state=random_state,
            class_weight="balanced",           # Fix
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state,
            class_weight="balanced",           # Fix
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=random_state,
            # GBM has no class_weight → use sample_weight in .fit() below
        ),
    }
    # ──────────────────────────────────────────────────────────────────────
 
    results = []
 
    for name, clf in classifiers.items():
        fold_metrics = {
            "AUC": [], "Accuracy": [], "F1": [],
            "Balanced_Accuracy_default": [],
            "Balanced_Accuracy_tuned":  [],
            "Threshold": [],
        }
 
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_tr_raw = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            X_te     = X.iloc[test_idx]  if hasattr(X, "iloc") else X[test_idx]
            y_tr_raw, y_te = y[train_idx], y[test_idx]
 
            # ── OPT A: SMOTE inside fold only ─────────────────────────────
            X_tr, y_tr = _apply_smote(X_tr_raw, y_tr_raw, random_state=random_state + fold_idx)
            # ──────────────────────────────────────────────────────────────
 
            # ── BUG 4 FIX for GBM: sample_weight in .fit() ───────────────
            if name == "Gradient Boosting":
                sw = compute_sample_weight("balanced", y_tr)
                clf.fit(X_tr, y_tr, sample_weight=sw)
            else:
                clf.fit(X_tr, y_tr)
            # ──────────────────────────────────────────────────────────────
 
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]
 
            # ── OPT B: threshold tuning via Youden's J ────────────────────
            best_thr = _best_threshold_youden(y_te, y_prob)
            y_pred_tuned = (y_prob >= best_thr).astype(int)
            # ──────────────────────────────────────────────────────────────
 
            fold_metrics["AUC"].append(roc_auc_score(y_te, y_prob))
            fold_metrics["Accuracy"].append(accuracy_score(y_te, y_pred))
            fold_metrics["F1"].append(f1_score(y_te, y_pred, zero_division=0))
            fold_metrics["Balanced_Accuracy_default"].append(
                balanced_accuracy_score(y_te, y_pred))
            fold_metrics["Balanced_Accuracy_tuned"].append(
                balanced_accuracy_score(y_te, y_pred_tuned))
            fold_metrics["Threshold"].append(best_thr)
 
        results.append({
            "Model":
                name,
            "AUC":
                f"{np.mean(fold_metrics['AUC']):.3f} ± {np.std(fold_metrics['AUC']):.3f}",
            "Accuracy":
                f"{np.mean(fold_metrics['Accuracy']):.3f}",
            "F1":
                f"{np.mean(fold_metrics['F1']):.3f}",
            "Balanced_Accuracy_default":
                f"{np.mean(fold_metrics['Balanced_Accuracy_default']):.3f}",
            "Balanced_Accuracy_tuned":
                f"{np.mean(fold_metrics['Balanced_Accuracy_tuned']):.3f}",
            "Threshold_mean":
                f"{np.mean(fold_metrics['Threshold']):.3f}",
        })
 
    print(f"[OK] Training complete for {X.shape[1]} features "
          f"(SMOTE={'on' if _SMOTE_AVAILABLE else 'off'}, threshold tuning=on).")
    return pd.DataFrame(results)