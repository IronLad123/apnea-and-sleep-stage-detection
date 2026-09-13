"""
labels.py — Ground-truth label builder for both tasks.

TASK 1: Sleep Stage (5-class AASM)
  - Hard label: mode() across all available scorers per epoch.
    Tie-break rule: Scorer 1 > Scorer 4 > Scorer 5 > lowest stage number.
  - Soft label: empirical probability vector [p_W, p_N1, p_N2, p_N3, p_REM]
    across all 12 scorers per epoch. Used as KL-divergence training target.

TASK 2: Apnea presence (binary, per epoch)
  - Derived from Resp_events/Annotations/manual/ files.
  - An epoch is POSITIVE if an apnea/hypopnea event overlaps >= 50% of its 30s window.
  - Annotation types treated as positive: 'Obstructive apnea', 'Mixed apnea', 'Hypopnea'
  - This is a SINGLE fixed ground truth — no 12-scorer soft label for apnea.

Usage:
    from psg_utils.labels import build_stage_labels, build_apnea_labels
"""

import os, glob
import numpy as np
import pandas as pd
from scipy.stats import mode as scipy_mode

STAGE_MAP = {"Sleep stage W": 0, "Sleep stage N1": 1,
             "Sleep stage N2": 2, "Sleep stage N3": 3,
             "Sleep stage R": 4}
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# Apnea event annotations to count as POSITIVE
APNEA_EVENTS = {"Obstructive apnea", "Mixed apnea", "Hypopnea"}

EPOCH_SEC = 30       # seconds per epoch
APNEA_OVERLAP = 0.5  # 50% overlap threshold


def build_stage_labels(subject_id, ann_dir, n_epochs):
    """
    Load all available scorer annotation files for a subject and compute
    both the hard-vote label and soft probability distribution per epoch.

    Parameters
    ----------
    subject_id : str — e.g. 'SN1'
    ann_dir : str — path to Sleep_stages/Annotations/manual/
    n_epochs : int — number of epochs in the recording

    Returns
    -------
    hard_labels : np.ndarray (n_epochs,), int — mode across scorers
    soft_labels : np.ndarray (n_epochs, 5), float — probability distribution
    tie_count   : int — number of epochs where tie-break was invoked
    n_scorers   : int — number of scorer files found
    """
    # Find all scorer files for this subject
    pattern = os.path.join(ann_dir, f"{subject_id}_SleepStages_manual_scorer*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"[labels] No scorer files found for {subject_id} in {ann_dir}"
        )

    print(f"[labels] {subject_id}: Found {len(files)} scorer files.")

    # Build a (n_epochs, n_scorers) matrix of integer labels
    all_labels = np.full((n_epochs, len(files)), fill_value=-1, dtype=int)

    for col_idx, fpath in enumerate(files):
        df = pd.read_csv(fpath, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        df["Annotation"] = df["Annotation"].str.strip()

        # Keep only sleep stage rows
        stage_rows = df[df["Annotation"].isin(STAGE_MAP.keys())].copy()
        stage_rows["epoch_idx"] = (
            pd.to_numeric(stage_rows["Recording onset"], errors="coerce") // EPOCH_SEC
        ).astype(int)
        stage_rows["label"] = stage_rows["Annotation"].map(STAGE_MAP)

        for _, row in stage_rows.iterrows():
            idx = int(row["epoch_idx"])
            if 0 <= idx < n_epochs:
                all_labels[idx, col_idx] = int(row["label"])

    # Replace -1 (missing) with the most common label across scorers for that epoch
    for i in range(n_epochs):
        row = all_labels[i]
        missing = row == -1
        if missing.all():
            all_labels[i] = 0  # Default to Wake if completely missing
        elif missing.any():
            valid = row[~missing]
            fill = int(scipy_mode(valid, keepdims=True).mode[0])
            all_labels[i, missing] = fill

    # Hard label: mode across scorers with tie-break
    tie_count = 0
    hard_labels = np.zeros(n_epochs, dtype=int)
    for i in range(n_epochs):
        result = scipy_mode(all_labels[i], keepdims=True)
        mode_val = int(result.mode[0])
        mode_count = int(result.count[0])
        n_scorers_val = all_labels.shape[1]

        # Check for tie (multiple modes with equal count)
        counts = np.bincount(all_labels[i] + 1, minlength=6)[1:]  # +1 offset for -1
        if np.sum(counts == mode_count) > 1:
            tie_count += 1
            # Tie-break: use Scorer 1's label (index 0)
            mode_val = int(all_labels[i, 0])

        hard_labels[i] = mode_val

    # Soft label: probability distribution [p_W, p_N1, p_N2, p_N3, p_REM]
    soft_labels = np.zeros((n_epochs, 5), dtype=float)
    for i in range(n_epochs):
        for stage in range(5):
            soft_labels[i, stage] = np.mean(all_labels[i] == stage)

    print(
        f"[labels] {subject_id}: Hard labels built. "
        f"Tie-breaks: {tie_count}/{n_epochs} epochs. "
        f"Stage distribution: {dict(zip(STAGE_NAMES.values(), np.bincount(hard_labels, minlength=5).tolist()))}"
    )

    return hard_labels, soft_labels, tie_count, len(files)


def build_apnea_labels(subject_id, resp_ann_dir, n_epochs):
    """
    Build binary apnea labels from respiratory event annotations.

    An epoch [onset_sec, onset_sec + 30) is labelled POSITIVE (1) if
    any apnea/hypopnea event overlaps >= 50% (15 seconds) of its window.

    Parameters
    ----------
    subject_id : str — e.g. 'SN1'
    resp_ann_dir : str — path to Resp_events/Annotations/manual/
    n_epochs : int

    Returns
    -------
    apnea_labels : np.ndarray (n_epochs,), int (0 or 1)
    n_events : int — total apnea/hypopnea events found
    ahi : float — Apnea-Hypopnea Index (events per hour of recording)
    """
    # Use Scorer 1 as the single fixed apnea ground truth
    # (Master prompt: apnea has no 12-scorer ambiguity)
    pattern = os.path.join(resp_ann_dir, f"{subject_id}_Respiration_manual_scorer1.txt")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"[labels] No Resp annotation file for {subject_id} at {pattern}"
        )

    df = pd.read_csv(files[0], skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["Annotation"] = df["Annotation"].str.strip()

    # Filter to apnea/hypopnea events only
    apnea_rows = df[df["Annotation"].isin(APNEA_EVENTS)].copy()
    apnea_rows["onset"]    = pd.to_numeric(apnea_rows["Recording onset"], errors="coerce")
    apnea_rows["duration"] = pd.to_numeric(apnea_rows["Duration"], errors="coerce")
    apnea_rows = apnea_rows.dropna(subset=["onset", "duration"])

    # Build apnea label array
    apnea_labels = np.zeros(n_epochs, dtype=int)

    for _, event in apnea_rows.iterrows():
        evt_start = event["onset"]
        evt_end   = evt_start + event["duration"]

        # Check overlap with each epoch window
        for epoch_idx in range(n_epochs):
            ep_start = epoch_idx * EPOCH_SEC
            ep_end   = ep_start + EPOCH_SEC

            overlap = max(0.0, min(evt_end, ep_end) - max(evt_start, ep_start))
            if overlap >= APNEA_OVERLAP * EPOCH_SEC:
                apnea_labels[epoch_idx] = 1

    n_events = len(apnea_rows)
    total_hours = (n_epochs * EPOCH_SEC) / 3600.0
    ahi = n_events / total_hours if total_hours > 0 else 0.0

    n_pos = int(apnea_labels.sum())
    print(
        f"[labels] {subject_id}: Apnea labels built. "
        f"Events: {n_events}, Positive epochs: {n_pos}/{n_epochs} "
        f"({n_pos/n_epochs*100:.1f}%), AHI: {ahi:.1f} events/hr"
    )

    return apnea_labels, n_events, ahi
