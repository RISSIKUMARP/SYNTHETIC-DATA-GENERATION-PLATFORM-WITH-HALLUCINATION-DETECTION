"""
Synthetic data generator for demos.

Produces statistically realistic credit-card transactions using the real
column distributions from creditcard.csv.  When the real CSV is absent it
falls back to hardcoded distribution parameters so the demo still runs.

Usage:
    python demo_data.py                  # 1000 rows, saves to data/synthetic/
    python demo_data.py --n 5000         # 5000 rows

Programmatic:
    from demo_data import generate_synthetic, generate_reference
    df = generate_synthetic(n=500, seed=42)
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_REAL_CSV = _PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
_SYNTH_DIR = _PROJECT_ROOT / "data" / "synthetic"

# ---------------------------------------------------------------------------
# Hardcoded distribution parameters (from EDA baseline)
# Used when creditcard.csv is not available
# ---------------------------------------------------------------------------

# Fraud-discriminative V-feature means for fraud rows
_FRAUD_V_MEANS = {
    "V1": -4.77, "V2": 3.58, "V3": -7.03, "V4": 4.54, "V5": -3.45,
    "V6": -1.64, "V7": -5.07, "V8": 1.35, "V9": -2.60, "V10": -5.68,
    "V11": 3.90, "V12": -6.26, "V13": -0.08, "V14": -6.97, "V15": -0.28,
    "V16": -4.49, "V17": -6.67, "V18": -1.81, "V19": 0.84, "V20": 0.39,
    "V21": 0.71, "V22": 0.28, "V23": -0.27, "V24": -0.05, "V25": 0.24,
    "V26": 0.20, "V27": 0.72, "V28": 0.35,
}

# Legit V-feature std-devs (PCA features, close to 1.0 for most)
_V_STDS = {
    "V1": 1.96, "V2": 1.65, "V3": 1.52, "V4": 1.42, "V5": 1.38,
    "V6": 1.33, "V7": 1.24, "V8": 1.19, "V9": 1.10, "V10": 1.09,
    "V11": 1.02, "V12": 1.00, "V13": 1.00, "V14": 0.96, "V15": 0.92,
    "V16": 0.88, "V17": 0.85, "V18": 0.84, "V19": 0.81, "V20": 0.77,
    "V21": 0.73, "V22": 0.73, "V23": 0.62, "V24": 0.61, "V25": 0.52,
    "V26": 0.48, "V27": 0.40, "V28": 0.33,
}

_V_COLS = [f"V{i}" for i in range(1, 29)]
_ALL_COLS = ["Time"] + _V_COLS + ["Amount", "Class"]


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def _load_real_stats():
    """Load stats from real data if available, else return None."""
    if not _REAL_CSV.exists():
        return None
    df = pd.read_csv(_REAL_CSV)
    stats = {}
    for col in _V_COLS:
        legit = df.loc[df["Class"] == 0, col]
        fraud = df.loc[df["Class"] == 1, col]
        stats[col] = {
            "legit_mean": float(legit.mean()),
            "legit_std": float(legit.std()),
            "fraud_mean": float(fraud.mean()),
            "fraud_std": float(fraud.std()),
        }
    stats["Amount"] = {
        "legit_log_mean": float(np.log1p(df.loc[df["Class"] == 0, "Amount"]).mean()),
        "legit_log_std": float(np.log1p(df.loc[df["Class"] == 0, "Amount"]).std()),
        "fraud_log_mean": float(np.log1p(df.loc[df["Class"] == 1, "Amount"]).mean()),
        "fraud_log_std": float(np.log1p(df.loc[df["Class"] == 1, "Amount"]).std()),
    }
    stats["Time"] = {
        "mean": float(df["Time"].mean()),
        "std": float(df["Time"].std()),
        "max": float(df["Time"].max()),
    }
    return stats


def _bootstrap_from_real(n: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    """
    Bootstrap synthetic data from the real CSV: sample with replacement
    and add small Gaussian noise to each numeric column.  This preserves
    inter-column correlations and passes KS tests much better than
    independent-column generation.
    """
    rng = np.random.default_rng(seed)
    df_real = pd.read_csv(_REAL_CSV)

    n_fraud = max(1, int(round(n * fraud_rate)))
    n_legit = n - n_fraud

    legit_pool = df_real[df_real["Class"] == 0]
    fraud_pool = df_real[df_real["Class"] == 1]

    legit_sample = legit_pool.sample(n=n_legit, replace=True, random_state=seed).reset_index(drop=True)
    fraud_sample = fraud_pool.sample(n=n_fraud, replace=True, random_state=seed).reset_index(drop=True)

    df = pd.concat([legit_sample, fraud_sample], ignore_index=True)

    # Add small noise to numeric columns so rows are not exact copies
    noise_scale = 0.02  # 2 % of column std
    for col in _V_COLS + ["Amount", "Time"]:
        col_std = df[col].std()
        noise = rng.normal(0, col_std * noise_scale, size=len(df))
        df[col] = df[col] + noise

    df["Amount"] = df["Amount"].clip(lower=0)
    df["Time"] = df["Time"].clip(lower=0)
    df["Class"] = df["Class"].astype(int)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df[_ALL_COLS]


def _generate_from_stats(n: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    """
    Fallback generator using hardcoded/loaded distribution parameters.
    Used when creditcard.csv is not available.
    """
    rng = np.random.default_rng(seed)
    stats = _load_real_stats()

    n_fraud = max(1, int(round(n * fraud_rate)))
    n_legit = n - n_fraud

    rows = np.zeros((n, len(_ALL_COLS)), dtype=np.float64)

    # --- Time ----------------------------------------------------------------
    if stats:
        t_mean, t_std, t_max = stats["Time"]["mean"], stats["Time"]["std"], stats["Time"]["max"]
    else:
        t_mean, t_std, t_max = 94814.0, 47488.0, 172792.0
    time_vals = rng.normal(t_mean, t_std, size=n)
    time_vals = np.clip(time_vals, 0, t_max)
    rows[:, 0] = time_vals

    # --- V-features ----------------------------------------------------------
    for idx, v in enumerate(_V_COLS, start=1):
        if stats:
            lm, ls = stats[v]["legit_mean"], stats[v]["legit_std"]
            fm, fs = stats[v]["fraud_mean"], stats[v]["fraud_std"]
        else:
            lm, ls = 0.0, _V_STDS[v]
            fm, fs = _FRAUD_V_MEANS[v], _V_STDS[v] * 1.5
        rows[:n_legit, idx] = rng.normal(lm, ls, size=n_legit)
        rows[n_legit:, idx] = rng.normal(fm, fs, size=n_fraud)

    # --- Amount (log-normal) -------------------------------------------------
    amount_idx = len(_ALL_COLS) - 2
    if stats:
        ll_mean, ll_std = stats["Amount"]["legit_log_mean"], stats["Amount"]["legit_log_std"]
        fl_mean, fl_std = stats["Amount"]["fraud_log_mean"], stats["Amount"]["fraud_log_std"]
    else:
        ll_mean, ll_std = 3.04, 1.95
        fl_mean, fl_std = 2.35, 1.75
    legit_amounts = np.clip(np.expm1(rng.normal(ll_mean, ll_std, size=n_legit)), 0, 25691.16)
    fraud_amounts = np.clip(np.expm1(rng.normal(fl_mean, fl_std, size=n_fraud)), 0, 2125.87)
    rows[:n_legit, amount_idx] = legit_amounts
    rows[n_legit:, amount_idx] = fraud_amounts

    # --- Class ---------------------------------------------------------------
    class_idx = len(_ALL_COLS) - 1
    rows[:n_legit, class_idx] = 0
    rows[n_legit:, class_idx] = 1

    df = pd.DataFrame(rows, columns=_ALL_COLS)
    df["Class"] = df["Class"].astype(int)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def generate_synthetic(n: int = 1000, fraud_rate: float = 0.00173, seed: int = 42) -> pd.DataFrame:
    """
    Generate *n* synthetic credit-card transactions.

    When creditcard.csv is available, uses bootstrap sampling with noise
    (preserves correlations, passes KS tests well).
    Otherwise falls back to independent-column generation from
    distribution parameters.

    Parameters
    ----------
    n : int
        Number of rows to generate.
    fraud_rate : float
        Target fraud rate (default 0.173 %).
    seed : int
        Numpy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns identical to creditcard.csv.
    """
    if _REAL_CSV.exists():
        return _bootstrap_from_real(n, fraud_rate, seed)
    return _generate_from_stats(n, fraud_rate, seed)


def generate_reference(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """
    Return a reference slice of the real data (for RAG comparison).

    If creditcard.csv exists, sample *n* rows from it.
    Otherwise, generate synthetic data as a stand-in.
    """
    if _REAL_CSV.exists():
        df = pd.read_csv(_REAL_CSV)
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return generate_synthetic(n=n, seed=seed + 999)


def save_demo_data(n: int = 1000, seed: int = 42) -> tuple:
    """Generate and persist synthetic + reference CSVs. Returns (synth_path, ref_path)."""
    _SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    synth_path = _SYNTH_DIR / "demo_synthetic.csv"
    ref_path = _SYNTH_DIR / "reference_500.csv"

    df_synth = generate_synthetic(n=n, seed=seed)
    df_synth.to_csv(synth_path, index=False)

    df_ref = generate_reference(n=500, seed=0)
    df_ref.to_csv(ref_path, index=False)

    return str(synth_path), str(ref_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demo synthetic data")
    parser.add_argument("--n", type=int, default=1000, help="Number of rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    t0 = time.perf_counter()
    sp, rp = save_demo_data(n=args.n, seed=args.seed)
    elapsed = time.perf_counter() - t0

    df = pd.read_csv(sp)
    print(f"Synthetic data : {sp}  ({len(df)} rows)")
    print(f"Reference data : {rp}")
    print(f"Fraud rate     : {df['Class'].mean():.4f} ({df['Class'].sum()} frauds)")
    print(f"Generated in   : {elapsed:.2f}s")
