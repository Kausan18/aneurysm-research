"""
models.py
Paper mapping: Section IV.D
Trains classifiers with stratified 5-fold CV, computes AUC, Accuracy, F1, Balanced Accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

def train_and_evaluate(X, y, cv_folds=5, random_state=42):
    """Train 3 classifiers with stratified CV and return metrics."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    }
    
    results = []
    for name, clf in classifiers.items():
        fold_metrics = {"AUC": [], "Accuracy": [], "F1": [], "Balanced_Accuracy": []}
        
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]
            
            fold_metrics["AUC"].append(roc_auc_score(y_te, y_prob))
            fold_metrics["Accuracy"].append(accuracy_score(y_te, y_pred))
            fold_metrics["F1"].append(f1_score(y_te, y_pred))
            fold_metrics["Balanced_Accuracy"].append(balanced_accuracy_score(y_te, y_pred))
            
        results.append({
            "Model": name,
            "AUC": f"{np.mean(fold_metrics['AUC']):.3f} ± {np.std(fold_metrics['AUC']):.3f}",
            "Accuracy": f"{np.mean(fold_metrics['Accuracy']):.3f}",
            "F1": f"{np.mean(fold_metrics['F1']):.3f}",
            "Balanced_Accuracy": f"{np.mean(fold_metrics['Balanced_Accuracy']):.3f}"
        })
        
    print(f"[OK] Training complete for {X.shape[1]} features.")
    return pd.DataFrame(results)