"""
quality.py — Per-epoch signal quality gating.

Runs BEFORE feature extraction on every epoch. Returns a boolean
quality mask. Flagged epochs are tracked but the DECISION on whether
to drop or down-weight is made ONCE per project (see QUALITY_ACTION).

Quality checks implemented:
  1. Flat-line detection   : rolling std ≈ 0 across the epoch.
  2. Saturation/clipping   : many samples pinned to ADC min/max.
  3. Amplitude outlier     : epoch RMS > Z_THRESH * subject's whole-night median RMS.

Usage:
    from psg_utils.quality import gate_epochs
    mask = gate_epochs(epochs, subject_id, signal_name)
    # mask.shape == (N_epochs,), dtype bool
    # True = GOOD epoch, False = FLAGGED
    clean_epochs = epochs[mask]
"""

import numpy as np

# --- Config (change these once, propagated everywhere) ---
QUALITY_ACTION = "drop"     # "drop" or "downweight"
FLAT_STD_THRESH  = 1e-6     # Below this std the channel is considered flat
SAT_FRAC_THRESH  = 0.05     # If >5% of samples are at ADC limits, flag as saturated
Z_THRESH         = 5.0      # Epoch RMS > 5× subject median → flag as artifact
MAX_DROP_RATE    = 0.20     # Warn if any subject loses >20% of epochs


def gate_epochs(epochs, subject_id, signal_name="signal"):
    """
    Evaluate signal quality for each 30-second epoch.

    Parameters
    ----------
    epochs : np.ndarray, shape (N_epochs, 7680)
    subject_id : str
    signal_name : str
        Label for logging (e.g. 'EEG', 'EMG chin').

    Returns
    -------
    mask : np.ndarray of bool, shape (N_epochs,)
        True = good epoch, False = flagged.
    report : dict
        Per-subject quality summary for output.md.
    """
    n = len(epochs)
    flags = np.ones(n, dtype=bool)  # Start: all good

    # 1. Flat-line detection
    epoch_stds = np.std(epochs, axis=1)
    flat_mask = epoch_stds < FLAT_STD_THRESH
    flags[flat_mask] = False

    # 2. Saturation/clipping
    # Assume ADC limits are min and max across the whole recording
    global_min = epochs.min()
    global_max = epochs.max()
    sat_frac = np.mean(
        (epochs == global_min) | (epochs == global_max),
        axis=1
    )
    sat_mask = sat_frac > SAT_FRAC_THRESH
    flags[sat_mask] = False

    # 3. Amplitude outlier (whole-night z-score on epoch RMS)
    epoch_rms = np.sqrt(np.mean(epochs ** 2, axis=1))
    median_rms = np.median(epoch_rms)
    if median_rms > 0:
        z_scores = epoch_rms / median_rms
        outlier_mask = z_scores > Z_THRESH
        flags[outlier_mask] = False
    else:
        outlier_mask = np.zeros(n, dtype=bool)

    n_flagged = int(np.sum(~flags))
    drop_rate = n_flagged / n

    report = {
        "subject":     subject_id,
        "signal":      signal_name,
        "n_total":     n,
        "n_flagged":   n_flagged,
        "drop_rate":   round(drop_rate, 4),
        "flat":        int(flat_mask.sum()),
        "saturated":   int(sat_mask.sum()),
        "outlier":     int(outlier_mask.sum()),
        "action":      QUALITY_ACTION,
    }

    print(
        f"[quality] {subject_id} {signal_name}: "
        f"{n_flagged}/{n} epochs flagged ({drop_rate*100:.1f}%). "
        f"Action: {QUALITY_ACTION}."
    )

    if drop_rate > MAX_DROP_RATE:
        print(
            f"[quality] ⚠️  WARNING {subject_id} {signal_name}: "
            f"Drop rate {drop_rate*100:.1f}% > {MAX_DROP_RATE*100:.0f}% threshold. "
            f"Check upstream EDF loading or electrode placement."
        )

    return flags, report
