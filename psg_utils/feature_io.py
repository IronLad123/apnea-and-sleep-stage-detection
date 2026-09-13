"""
feature_io.py — Standardized Parquet feature caching.

All signal notebooks MUST call save_features() at the end.
Multimodal fusion (Notebook 6) loads all signals by calling load_all_features(),
which performs a deterministic join on (subject_id, epoch_idx).

Features are stored at:
    features/{signal_name}/{subject_id}.parquet

Usage:
    from psg_utils.feature_io import save_features, load_features, load_all_features
    save_features(df, signal_name='eeg', subject_id='SN1')
    df_eeg = load_features('eeg', 'SN1')
    df_all = load_all_features(['eeg', 'eog', 'emg', 'ecg', 'resp_sao2'], subjects=['SN1',...])
"""

import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")


def save_features(df, signal_name, subject_id):
    """
    Save a feature DataFrame to Parquet format.

    The DataFrame MUST contain columns:
      - 'subject'   : subject ID string (e.g. 'SN1')
      - 'epoch_idx' : integer epoch index (0-based)
      - 'hard_label': integer sleep stage (0-4), may also have 'apnea_label'
      - All feature columns

    Parameters
    ----------
    df : pd.DataFrame
    signal_name : str — e.g. 'eeg', 'eog', 'emg', 'ecg', 'resp_sao2'
    subject_id : str — e.g. 'SN1'
    """
    # Validate required columns
    required = {"subject", "epoch_idx", "hard_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[feature_io] DataFrame missing required columns: {missing}"
        )

    out_dir = os.path.join(FEATURES_DIR, signal_name)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{subject_id}.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(
        f"[feature_io] Saved {len(df)} epochs × {len(df.columns)} cols "
        f"→ features/{signal_name}/{subject_id}.parquet"
    )


def load_features(signal_name, subject_id):
    """Load a single subject's feature Parquet for a given signal."""
    path = os.path.join(FEATURES_DIR, signal_name, f"{subject_id}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[feature_io] No Parquet found at {path}. "
            f"Run the {signal_name.upper()} signal notebook first."
        )
    return pd.read_parquet(path, engine="pyarrow")


def load_all_features(signal_names, subjects):
    """
    Load and merge feature tables for all signals and subjects.

    Performs an INNER JOIN on (subject, epoch_idx) — both must match exactly.
    This ensures epoch alignment across all modalities.

    Parameters
    ----------
    signal_names : list of str — e.g. ['eeg', 'eog', 'emg', 'ecg', 'resp_sao2']
    subjects : list of str — e.g. ['SN1', 'SN2', 'SN3', 'SN4', 'SN5']

    Returns
    -------
    pd.DataFrame — merged multimodal feature table
    """
    all_dfs = []

    for subject_id in subjects:
        subject_dfs = {}
        for signal in signal_names:
            df = load_features(signal, subject_id)
            # Strip label columns except from the first signal (avoid duplicates)
            if signal != signal_names[0]:
                label_cols = ["hard_label", "apnea_label", "soft_label", "subject"]
                df = df.drop(columns=[c for c in label_cols if c in df.columns], errors="ignore")
            subject_dfs[signal] = df

        # Merge all signals on (subject, epoch_idx)
        merged = subject_dfs[signal_names[0]]
        for signal in signal_names[1:]:
            right = subject_dfs[signal]
            merged = merged.merge(right, on=["epoch_idx"], how="inner", suffixes=("", f"_{signal}"))

        all_dfs.append(merged)

    final = pd.concat(all_dfs, ignore_index=True)
    print(
        f"[feature_io] Loaded {len(signal_names)} signals × {len(subjects)} subjects "
        f"→ {len(final)} epochs × {len(final.columns)} features."
    )
    return final
