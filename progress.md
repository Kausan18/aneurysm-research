# 🚀 Project Progress Tracker

## Phase 1: Baseline Reproduction
- [x] ✅ Folder structure & VS Code setup
- [x] ✅ Config & nominal parameters defined
- [x] ✅ Data loading & merging (`data_loader.py`)
- [x] ✅ Preprocessing: imputation, normalization, L/H/S surrogates (`preprocessing.py`)
- [x] ✅ ODE simulation & biomarker extraction (`ode_model.py`)
- [x] ✅ Feature set construction (baseline/ODE/hybrid) (`features.py`)
- [x] ✅ Classifier training & 5-fold CV (`models.py`)
- [x] ✅ Visualization & results export (`visualization.py`)
- [x] ✅ Pipeline orchestration (`main.py`)
- [ ] ⏳ Run baseline & record initial metrics

## Phase 2: Coefficient Optimization (Next)
- [ ] Define search space for weights & ODE params
- [ ] Implement grid/Bayesian optimization loop
- [ ] Nested CV to prevent data leakage
- [ ] Statistical significance testing (paired t-test)
- [ ] Compare optimized vs baseline results
- [ ] Document sensitivity analysis & clinical interpretability

## Notes & Observations
- LSA not present in dataset → set `LSA_n = 0` temporarily
- All nominal coefficients from paper Sec II.E/F
- Metrics: AUC, Accuracy, F1, Balanced Accuracy
- Random seed: 42 for reproducibility