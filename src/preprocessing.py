"""Parse pulse recordings and extract fixed-length waveform windows.

The packet layout and channel selection follow the supplied gru_exm.py.
"""

from pathlib import Path
import warnings

import numpy as np
from scipy.signal import find_peaks_cwt, savgol_filter


def read_pulse_file(path):
    """Decode valid 19-byte hexadecimal frames into [time, 3] signals.

    Each nonempty line contains 19 hexadecimal byte tokens. Bytes 4:18
    contribute to the checksum in byte 18. The first 12 payload bytes
    contain six little-endian values; channels 1, 3, and 5 are retained.
    Indices above are zero-based, matching the original script.
    """
    path = Path(path)
    samples = []
    invalid_frames = 0

    with path.open("r", encoding="utf-8-sig") as recording:
        for line in recording:
            tokens = line.split()
            if not tokens:
                continue
            try:
                if len(tokens) != 19:
                    raise ValueError("Expected 19 bytes")
                values = [int(token, 16) for token in tokens]
                if any(value < 0 or value > 255 for value in values):
                    raise ValueError("Byte outside 0-255")
                packet = np.asarray(values, dtype=np.int64)
            except ValueError:
                invalid_frames += 1
                continue

            if int(packet[4:18].sum()) % 256 != packet[18]:
                invalid_frames += 1
                continue

            payload = packet[4:16]
            channels = payload[0::2] + 256 * payload[1::2]
            samples.append(channels[[1, 3, 5]])

    if invalid_frames:
        warnings.warn(
            f"Skipped {invalid_frames} invalid frames in {path.name}.",
            UserWarning,
            stacklevel=2,
        )
    if not samples:
        raise ValueError(f"No valid pulse frames found in {path}.")

    # Floating point prevents smoothed values from being truncated to integers.
    return np.asarray(samples, dtype=np.float32)


def extract_windows(signals, points=500, peak_width=20):
    """Return normalized pulse windows with shape [windows, points, 3].

    This preserves the original six-peak-interval windows, edge exclusion,
    and model-input channel order (2, 0, 1). Windows overlap: split whole
    recordings, or subjects when identified, before creating dataset splits.
    """
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 2 or signals.shape[1] != 3:
        raise ValueError("Signals must have shape [time, 3].")
    if points < 2 or peak_width <= 0:
        raise ValueError("points must be >= 2 and peak_width must be > 0.")
    if not np.isfinite(signals).all():
        raise ValueError("Signals contain non-finite values.")

    empty = np.empty((0, points, 3), dtype=np.float32)
    if len(signals) < 5:
        return empty

    filtered = savgol_filter(signals, 5, 3, axis=0, mode="nearest")
    peaks = np.asarray(
        find_peaks_cwt(filtered[:, 0], widths=[peak_width]), dtype=np.int64
    )
    windows = []

    # Same bounds as the original: peaks[i + 1] through peaks[i + 7].
    for index in range(len(peaks) - 10):
        start, stop = peaks[index + 1], peaks[index + 7]
        segment = filtered[start:stop, :][:, [2, 0, 1]]
        if len(segment) < 2:
            continue

        source_time = np.arange(len(segment))
        target_time = np.linspace(0, len(segment) - 1, points)
        window = np.column_stack(
            [np.interp(target_time, source_time, segment[:, c]) for c in range(3)]
        )
        minimum = window.min(axis=0, keepdims=True)
        span = window.max(axis=0, keepdims=True) - minimum
        window = (window - minimum) / np.maximum(span, 1e-8)
        windows.append(window.astype(np.float32))

    return np.stack(windows) if windows else empty
