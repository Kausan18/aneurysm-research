"""
models.py
Paper mapping: Section IV.D

Phase 1 settings:
- USE_CALIBRATION = False   <-- Phase 2 feature only; OFF for clean Phase 1
- USE_SMOTE       = False   <-- disabled (44 minority / ~7 per fold is too few)

Classifiers: LR (balanced), RF (balanced, no calibration), GBM (sample_weight)
CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=3) -> 15 folds
Threshold tuning: Youden's J per fold
Metrics: AUC +- std, Accuracy, F1, Balanced_Accuracy_default,
         Balanced_Accuracy_tuned, Threshold_mean

NOTE FOR PHASE 2: set USE_CALIBRATION = True before running main_phase2.py
to re-enable isotonic RF calibration (fixes inf thresholds on small datasets).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    balanced_accuracy_score, roc_curve,
)

# SMOTE: disabled (too few minority samples per fold on n=103)
USE_SMOTE = False

# RF calibration: OFF for Phase 1.
# Set True in Phase 2 to fix inf thresholds via CalibratedClassifierCV(isotonic).
USE_CALIBRATION = False

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False
    if USE_SMOTE:
        print("[WARN] imbalanced-learn not installed -- SMOTE skipped.")


def _apply_smote(X_tr, y_tr, random_state=42):
    if not USE_SMOTE or not _SMOTE_AVAILABLE:
        return X_tr, y_tr
    minority_count = int(np.sum(y_tr == 1))
    k = min(3, minority_count - 1)
    if k < 1:
        return X_tr, y_tr
    sm = SMOTE(k_neighbors=k, random_state=random_state)
    return sm.fit_resample(X_tr, y_tr)


def _best_threshold_youden(y_true, y_prob):
    """Return threshold maximising Youden's J = Sensitivity + Specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def train_and_evaluate(X, y, n_splits=5, n_repeats=3, random_state=42):
    """Train 3 classifiers with RepeatedStratifiedKFold and return metrics DataFrame.

    Parameters
    ----------
    X            : pd.DataFrame or np.ndarray
    y            : np.ndarray (int) -- binary labels
    n_splits     : int -- inner fold count (default 5)
    n_repeats    : int -- repetitions (default 3) -> 15 total evaluations
    random_state : int

    Returns
    -------
    pd.DataFrame with columns:
        Model, AUC, Accuracy, F1,
        Balanced_Accuracy_default, Balanced_Accuracy_tuned, Threshold_mean
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    _rf_base = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight="balanced",
    )
    _rf_clf = (
        CalibratedClassifierCV(_rf_base, method="isotonic", cv=3)
        if USE_CALIBRATION
        else _rf_base
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Random Forest": _rf_clf,
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state,
        ),
    }

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

            X_tr, y_tr = _apply_smote(
                X_tr_raw, y_tr_raw, random_state=random_state + fold_idx
            )

            if name == "Gradient Boosting":
                sw = compute_sample_weight("balanced", y_tr)
                clf.fit(X_tr, y_tr, sample_weight=sw)
            else:
                clf.fit(X_tr, y_tr)

            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]

            auc_val = roc_auc_score(y_te, y_prob)
            if auc_val < 0.50:
                print(f"[WARN] {name} fold {fold_idx}: AUC={auc_val:.3f} < 0.50")

            best_thr     = _best_threshold_youden(y_te, y_prob)
            y_pred_tuned = (y_prob >= best_thr).astype(int)

            fold_metrics["AUC"].append(auc_val)
            fold_metrics["Accuracy"].append(accuracy_score(y_te, y_pred))
            fold_metrics["F1"].append(f1_score(y_te, y_pred, zero_division=0))
            fold_metrics["Balanced_Accuracy_default"].append(
                balanced_accuracy_score(y_te, y_pred)
            )
            fold_metrics["Balanced_Accuracy_tuned"].append(
                balanced_accuracy_score(y_te, y_pred_tuned)
            )
            fold_metrics["Threshold"].append(best_thr)

        total_folds = n_splits * n_repeats
        results.append({
            "Model": name,
            "AUC":
                f"{np.mean(fold_metrics['AUC']):.3f} +- "
                f"{np.std(fold_metrics['AUC']):.3f}",
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
        print(f"[OK] {name}: AUC="
              f"{np.mean(fold_metrics['AUC']):.3f} "
              f"(+-{np.std(fold_metrics['AUC']):.3f}, "
              f"{total_folds} folds)")

    smote_status = "on" if (USE_SMOTE and _SMOTE_AVAILABLE) else "off"
    cal_status   = "on" if USE_CALIBRATION else "off"
    n_feat = X.shape[1] if hasattr(X, "shape") else "?"
    print(f"[OK] Training complete for {n_feat} features "
          f"(SMOTE={smote_status}, RF_calibration={cal_status}, "
          f"threshold_tuning=on, {n_splits}x{n_repeats} RepeatedStratifiedKFold).")
    return pd.DataFrame(results)