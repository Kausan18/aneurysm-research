"""
data_loader.py
Paper mapping: Section IV.A

Loads Merged_Aneurysm.csv (103 patients, 44R / 59U).
- Encodes target:   ruptureStatus  R=1, U=0
- Encodes location: one-hot (ICA/MCA/ACA/BAS)
- Encodes type:     binary TER=1, LAT=0
- Encodes sex:      binary F=1, M=0
- Drops CFD columns (95/103 missing — do NOT impute)
- Drops ID / duplicate columns
"""

import os
import pandas as pd
from config import DATA_DIR, CFD_DROP_COLS, ID_DROP_COLS


def load_data():
    """Load and clean Merged_Aneurysm.csv.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with:
        - `y` column (int): 1 = ruptured, 0 = unruptured
        - All geometry / morphological features present
        - CFD and ID columns removed
        - Categorical features encoded
    y  : np.ndarray (int, shape [103])
        Binary rupture label.
    """
    csv_path = os.path.join(DATA_DIR, "Merged_Aneurysm.csv")
    df = pd.read_csv(csv_path)

    print(f"[OK] Loaded {csv_path}: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── 1. Drop CFD columns (95/103 missing — cannot impute) ─────────────────
    cols_to_drop = [c for c in CFD_DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[OK] Dropped {len(cols_to_drop)} CFD columns.")

    # ── 2. Drop ID / duplicate columns ───────────────────────────────────────
    id_cols_found = [c for c in ID_DROP_COLS if c in df.columns]
    df.drop(columns=id_cols_found, inplace=True)
    print(f"[OK] Dropped {len(id_cols_found)} ID/duplicate columns.")

    # ── 3. Encode target: R=1, U=0 ───────────────────────────────────────────
    if df["ruptureStatus"].dtype == object:
        df["y"] = (df["ruptureStatus"] == "R").astype(int)
    else:
        df["y"] = df["ruptureStatus"].astype(int)
    df.drop(columns=["ruptureStatus"], inplace=True)

    r_count = int(df["y"].sum())
    u_count = int((df["y"] == 0).sum())
    print(f"[OK] Target encoded: {r_count} ruptured (1), {u_count} unruptured (0).")

    # ── 4. Encode categorical features ───────────────────────────────────────
    # sex: F=1, M=0 (keep as single binary column)
    if "sex" in df.columns:
        df["sex"] = (df["sex"].str.strip().str.upper() == "F").astype(int)

    # aneurysmType: TER=1, LAT=0
    if "aneurysmType" in df.columns:
        df["aneurysmType"] = (df["aneurysmType"].str.strip().str.upper() == "TER").astype(int)

    # aneurysmLocation: one-hot encode (ICA / MCA / ACA / BAS)
    if "aneurysmLocation" in df.columns:
        loc_dummies = pd.get_dummies(
            df["aneurysmLocation"].str.strip().str.upper(),
            prefix="loc",
            drop_first=False,   # keep all 4 levels for interpretability
        ).astype(int)
        df.drop(columns=["aneurysmLocation"], inplace=True)
        df = pd.concat([df, loc_dummies], axis=1)

    # multipleAneurysms: convert Y/N → 1/0 if present
    if "multipleAneurysms" in df.columns:
        if df["multipleAneurysms"].dtype == object:
            df["multipleAneurysms"] = (
                df["multipleAneurysms"].str.strip().str.upper() == "Y"
            ).astype(int)

    y = df["y"].values
    print(f"[OK] Data loading complete. Final shape: {df.shape}")
    return df, y


if __name__ == "__main__":
    df, y = load_data()
    print(df.dtypes)
    print(df.head())