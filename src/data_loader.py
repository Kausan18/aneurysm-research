"""
data_loader.py
Paper mapping: Section IV.A

Loads Merged_Aneurysm.csv (103 patients, 44R / 59U) and joins with Aneurysm_WSS_values_clean.csv.
- Encodes target:   ruptureStatus  R=1, U=0
- Encodes location: one-hot (ICA/MCA/ACA/BAS)
- Encodes type:     binary TER=1, LAT=0
- Encodes sex:      binary F=1, M=0
- Drops CFD columns (95/103 missing -- do NOT impute)
- Drops ID / duplicate columns
- Joins with Aneurysm_WSS_values_clean.csv on case_id
- Renames WSS_mean_dyn_cm2 to WSS_mean for use as H surrogate

FIX: ruptureStatus is always a string dtype ('R'/'U') in this CSV.
     Previous code had an else-branch that tried .astype(int) directly on
     the string column, causing a ValueError at runtime. Now always maps
     via .map({"R": 1, "U": 0}) regardless of dtype.
"""

import os
import pandas as pd
from config import DATA_DIR, CFD_DROP_COLS, ID_DROP_COLS


def load_data():
    """Load and clean Merged_Aneurysm.csv, then join with WSS data.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with:
        - `y` column (int): 1 = ruptured, 0 = unruptured
        - All geometry / morphological features present
        - WSS_mean column (from Aneurysm_WSS_values_clean.csv)
        - CFD and ID columns removed
        - Categorical features encoded
    y  : np.ndarray (int, shape [103])
        Binary rupture label.
    """
    # Load main dataset
    csv_path = os.path.join(DATA_DIR, "Merged_Aneurysm.csv")
    df = pd.read_csv(csv_path)

    print(f"[OK] Loaded {csv_path}: {df.shape[0]} rows x {df.shape[1]} columns")

    # Load WSS data and join
    wss_path = os.path.join(DATA_DIR, "Aneurysm_WSS_values_clean.csv")
    wss_df = pd.read_csv(wss_path)
    print(f"[OK] Loaded {wss_path}: {wss_df.shape[0]} rows x {wss_df.shape[1]} columns")

    # Inner join on case_id
    df = pd.merge(df, wss_df[["case_id", "WSS_mean_dyn_cm2"]], on="case_id", how="inner")
    print(f"[OK] Joined on case_id: {df.shape[0]} rows after inner join")

    # Rename WSS column for clarity
    df.rename(columns={"WSS_mean_dyn_cm2": "WSS_mean"}, inplace=True)
    print(f"[OK] Renamed WSS_mean_dyn_cm2 to WSS_mean")

    # Convert WSS from dyn/cm^2 to Pascal (1 dyn/cm^2 = 0.1 Pa)
    df["WSS_mean"] = df["WSS_mean"] * 0.1
    print(f"[OK] Converted WSS_mean from dyn/cm^2 to Pascal (1 dyn/cm^2 = 0.1 Pa)")
    print(f"     WSS_mean range: {df['WSS_mean'].min():.4f} to {df['WSS_mean'].max():.4f} Pa")

    # 1. Drop CFD columns (95/103 missing -- cannot impute)
    cols_to_drop = [c for c in CFD_DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[OK] Dropped {len(cols_to_drop)} CFD columns.")

    # 2. Drop ID / duplicate columns
    id_cols_found = [c for c in ID_DROP_COLS if c in df.columns]
    df.drop(columns=id_cols_found, inplace=True)
    print(f"[OK] Dropped {len(id_cols_found)} ID/duplicate columns.")

    # 3. Encode target: R=1, U=0  (always string in this CSV)
    df["y"] = (
        df["ruptureStatus"]
        .astype(str)
        .str.strip()
        .map({"R": 1, "U": 0})
        .astype(int)
    )
    df.drop(columns=["ruptureStatus"], inplace=True)

    r_count = int(df["y"].sum())
    u_count = int((df["y"] == 0).sum())
    print(f"[OK] Target encoded: {r_count} ruptured (1), {u_count} unruptured (0).")

    # 4. Encode categorical features
    if "sex" in df.columns:
        df["sex"] = (df["sex"].str.strip().str.upper() == "F").astype(int)

    if "aneurysmType" in df.columns:
        df["aneurysmType"] = (
            df["aneurysmType"].str.strip().str.upper() == "TER"
        ).astype(int)

    if "aneurysmLocation" in df.columns:
        loc_dummies = pd.get_dummies(
            df["aneurysmLocation"].str.strip().str.upper(),
            prefix="loc",
            drop_first=False,
        ).astype(int)
        df.drop(columns=["aneurysmLocation"], inplace=True)
        df = pd.concat([df, loc_dummies], axis=1)

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