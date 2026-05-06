"""
data_loader.py
Paper mapping: Section IV.A
Loads CSVs, cleans column names, merges on patient ID, filters aneurysm patients with rupture labels.
"""

import pandas as pd
import os
from config import DATA_DIR, COLUMN_MAP

def load_and_merge_data():
    """Load clinical, morphological, and hemodynamic CSVs and merge them."""
    # Load files
    clinical = pd.read_csv(os.path.join(DATA_DIR, "clinical_all.csv"))
    morph = pd.read_csv(os.path.join(DATA_DIR, "morphological_aneurysm_artery.csv"))
    hemo = pd.read_csv(os.path.join(DATA_DIR, "hemodynamic_aneurysm_artery.csv"))
    
    # Clean column names (remove spaces, brackets, special chars)
    clean_col = lambda c: c.strip().replace("\n", "_").replace(" ", "_").replace("(", "").replace(")", "").replace("（", "").replace("）", "").replace("[", "").replace("]", "").replace(",", "").replace(".", "_")
    morph.columns = [clean_col(c) for c in morph.columns]
    hemo.columns = [clean_col(c) for c in hemo.columns]
    clinical.columns = [clean_col(c) for c in clinical.columns]
    
    # Select relevant columns (use clean_col to build expected column names)
    morph_cols = ["number", clean_col(COLUMN_MAP["morph"]["AR"]),
                  clean_col(COLUMN_MAP["morph"]["SR"])]
    hemo_cols = ["number", clean_col(COLUMN_MAP["hemo"]["WSS"]),
                 clean_col(COLUMN_MAP["hemo"]["OSI"])]
    clinical_cols = ["number", clean_col(COLUMN_MAP["clinical"]["HAS_ANEURYSM"]),
                     clean_col(COLUMN_MAP["clinical"]["RUPTURE"])]
    
    # Merge
    df = clinical[clinical_cols].merge(morph[morph_cols], on="number", how="inner")
    df = df.merge(hemo[hemo_cols], on="number", how="inner")
    
    # Filter: has aneurysm + valid rupture label
    has_aneurysm_col = clean_col(COLUMN_MAP["clinical"]["HAS_ANEURYSM"])
    rupture_col = clean_col(COLUMN_MAP["clinical"]["RUPTURE"])
    # Convert to string for comparison since CSV contains string values
    df[has_aneurysm_col] = df[has_aneurysm_col].astype(str)
    df[rupture_col] = df[rupture_col].astype(str)
    df = df[(df[has_aneurysm_col] == "1") & (df[rupture_col] != "nan") & (df[rupture_col].notna())]
    # Convert rupture to int for downstream processing
    df[rupture_col] = pd.to_numeric(df[rupture_col], errors='coerce').astype(int)
    
    print(f"[OK] Loaded {len(df)} patients with aneurysms and rupture labels.")
    return df

if __name__ == "__main__":
    df = load_and_merge_data()
    print(df.head())