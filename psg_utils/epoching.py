"""
epoching.py — Strict 30-second epoch segmentation at 256 Hz.

This is the ground-truth epoch engine for ALL signal notebooks.
Every signal notebook must import and use `make_epochs()` — never
roll your own epoch loop that might accidentally differ in length or alignment.

Usage:
    from psg_utils.epoching import make_epochs
    signal = raw.get_data(picks=[channel_name])[0]  # 1D numpy array
    epochs = make_epochs(signal, subject_id)
    # epochs.shape == (N_epochs, 7680)
"""

import numpy as np

FS         = 256          # Sampling frequency (Hz) — asserted, not assumed
EPOCH_SEC  = 30           # Epoch length in seconds (AASM standard)
EPOCH_SAMP = FS * EPOCH_SEC  # = 7680 samples per epoch


def assert_fs(raw, subject_id):
    """Hard-assert that the EDF sampling frequency is exactly 256 Hz."""
    actual = raw.info["sfreq"]
    assert actual == float(FS), (
        f"[epoching] {subject_id}: Expected fs={FS} Hz, got {actual} Hz. "
        f"Check EDF header — this is not the PSG-IPA dataset."
    )
    print(f"[epoching] {subject_id}: fs={actual} Hz ✓")


def make_epochs(signal_1d, subject_id, fs=FS, epoch_sec=EPOCH_SEC):
    """
    Segment a 1D signal array into non-overlapping 30-second epochs.

    Truncates the signal to an integer multiple of epoch length:
        N_epochs = floor(L / 7680)
    The last incomplete epoch is silently discarded (standard PSG practice).

    Parameters
    ----------
    signal_1d : np.ndarray, shape (L,)
        Raw signal samples for a single channel.
    subject_id : str
        For informative logging only.
    fs : int
        Sampling frequency (Hz). Must equal 256.
    epoch_sec : int
        Epoch duration in seconds (must be 30).

    Returns
    -------
    np.ndarray, shape (N_epochs, 7680)
        Each row is one 30-second epoch.
    """
    epoch_samp = fs * epoch_sec
    n_epochs = len(signal_1d) // epoch_samp

    # Truncate to clean multiple
    signal_trimmed = signal_1d[:n_epochs * epoch_samp]
    epochs = signal_trimmed.reshape(n_epochs, epoch_samp)

    print(
        f"[epoching] {subject_id}: {len(signal_1d)} samples → "
        f"{n_epochs} epochs × {epoch_samp} samples "
        f"(= {n_epochs * epoch_sec / 60:.1f} min)"
    )
    return epochs


def epoch_index_array(n_epochs):
    """Return a 0-indexed array of epoch indices: [0, 1, 2, ..., N-1]."""
    return np.arange(n_epochs)
