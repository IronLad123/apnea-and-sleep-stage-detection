"""
splits.py — Subject-level Leave-One-Subject-Out (LOSO) generator.

Zero data leakage guaranteed by a hard assert on every split.
All 5 signal notebooks must use this — no ad-hoc train/test splits.

Usage:
    from psg_utils.splits import loso_splits
    for fold, (train_df, test_df, test_subject) in enumerate(loso_splits(df)):
        ...
"""

import numpy as np
import pandas as pd


def loso_splits(df, subject_col="subject"):
    """
    Yield Leave-One-Subject-Out train/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a `subject` column (e.g. 'SN1', 'SN2', ...).
    subject_col : str
        Column name identifying the subject per row.

    Yields
    ------
    train_df : pd.DataFrame — all subjects except the held-out one
    test_df  : pd.DataFrame — the single held-out subject
    test_subject : str — which subject was held out
    """
    subjects = sorted(df[subject_col].unique())
    print(f"[splits] LOSO over {len(subjects)} subjects: {subjects}")

    for test_subject in subjects:
        train_df = df[df[subject_col] != test_subject].copy()
        test_df  = df[df[subject_col] == test_subject].copy()

        # === MANDATORY LEAKAGE ASSERT ===
        overlap = set(train_df[subject_col]) & set(test_df[subject_col])
        assert len(overlap) == 0, (
            f"[splits] DATA LEAKAGE DETECTED! "
            f"Subject(s) {overlap} appear in both train and test sets."
        )

        print(
            f"[splits] Fold: test={test_subject} "
            f"| train={len(train_df)} epochs "
            f"| test={len(test_df)} epochs"
        )

        yield train_df, test_df, test_subject


def get_X_y(df, feature_cols, label_col="hard_label"):
    """
    Extract feature matrix X and label vector y from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list of str — column names for features
    label_col : str — 'hard_label' or 'apnea_label'

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(int)
    return X, y


def make_sequences(X, y, seq_len=5):
    """
    Convert a 2D feature matrix into overlapping sequences for GRU/LSTM.
    Used for sequential modeling (one data point per epoch, context from neighbors).

    Parameters
    ----------
    X : np.ndarray (N, n_features)
    y : np.ndarray (N,)
    seq_len : int — number of consecutive epochs per sequence

    Returns
    -------
    X_seq : np.ndarray (N - seq_len + 1, seq_len, n_features)
    y_seq : np.ndarray (N - seq_len + 1,) — label of the LAST epoch in the window
    """
    X_seq, y_seq = [], []
    for i in range(seq_len - 1, len(X)):
        X_seq.append(X[i - seq_len + 1 : i + 1])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=int)
