"""
channel_resolver.py — Resolve electrode name discrepancies across subjects.

SN1 Sleep_stages EDF has 'EEG Cz-M1' (midline central).
SN2-SN5 Sleep_stages EDFs have 'EEG C4-M1' (right central).
Both are valid AASM Central derivations. This module handles the fallback
and logs the resolved channel name per subject for full traceability.

Usage:
    from psg_utils.channel_resolver import resolve_eeg_central
    ch = resolve_eeg_central(raw, subject_id)
"""

import os, json
from datetime import datetime

# Priority order for Central EEG derivations (most to least preferred)
EEG_CENTRAL_PRIORITY = ["EEG Cz-M1", "EEG C4-M1", "EEG F4-M1"]

# Priority order for EOG channels
EOG_CHANNELS = ["EOG E1-M2", "EOG E2-M2"]

# EMG channels
EMG_CHIN = "EMG chin"
EMG_LAT  = "EMG LAT"
EMG_RAT  = "EMG RAT"

# Other
ECG_CHANNEL   = "ECG"
RESP_CHANNELS = ["Resp nasal", "Resp chest", "Resp abdomen"]
SAO2_CHANNEL  = "SaO2"


def resolve_eeg_central(raw, subject_id, log_dir=None):
    """
    Find the best available Central EEG channel in a raw MNE object.
    Writes the resolved channel to a JSON log file if log_dir is provided.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded EDF file.
    subject_id : str
        e.g. 'SN1'.
    log_dir : str or None
        Directory to write channel_log.json. Defaults to project root.

    Returns
    -------
    str : The resolved channel name (e.g. 'EEG Cz-M1' or 'EEG C4-M1').
    """
    available = raw.ch_names
    for ch in EEG_CENTRAL_PRIORITY:
        if ch in available:
            _log_resolved(subject_id, "EEG_central", ch, log_dir)
            print(f"[channel_resolver] {subject_id}: Using '{ch}' for Central EEG.")
            return ch

    raise ValueError(
        f"[channel_resolver] {subject_id}: No Central EEG channel found. "
        f"Available channels: {available}"
    )


def resolve_channels(raw, subject_id, log_dir=None):
    """
    Resolve and verify ALL expected channels in the EDF.
    Returns a dict: {'eeg': ch_name, 'eog': [ch1, ch2], 'emg_chin': ch, ...}
    """
    available = set(raw.ch_names)
    resolved = {}

    resolved["eeg"] = resolve_eeg_central(raw, subject_id, log_dir)

    # EOG
    eog = [ch for ch in EOG_CHANNELS if ch in available]
    if len(eog) < 2:
        print(f"[channel_resolver] WARNING {subject_id}: Only {len(eog)}/2 EOG channels found.")
    resolved["eog"] = eog

    # EMG
    resolved["emg_chin"] = EMG_CHIN if EMG_CHIN in available else None
    resolved["emg_lat"]  = EMG_LAT  if EMG_LAT  in available else None
    resolved["emg_rat"]  = EMG_RAT  if EMG_RAT  in available else None

    # ECG
    resolved["ecg"] = ECG_CHANNEL if ECG_CHANNEL in available else None

    # Respiration
    resolved["resp"] = [ch for ch in RESP_CHANNELS if ch in available]

    # SaO2
    resolved["sao2"] = SAO2_CHANNEL if SAO2_CHANNEL in available else None

    _log_resolved(subject_id, "all_channels", resolved, log_dir)
    return resolved


def _log_resolved(subject_id, key, value, log_dir):
    """Append resolved channel info to channel_resolution_log.json."""
    if log_dir is None:
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    log_path = os.path.join(log_dir, "channel_resolution_log.json")

    # Load existing log or create new
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
    else:
        log = {}

    if subject_id not in log:
        log[subject_id] = {}

    log[subject_id][key] = value
    log[subject_id]["timestamp"] = datetime.now().isoformat()

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
