# ./src/euler_disk/data_processing.py
import json

import numpy as np
from scipy import signal
from scipy.io import wavfile


def short_time_ft(
    input_path: str, output_dir: str, window_size: int, hop_size: int
) -> None:
    rate, data = wavfile.read(input_path)

    # Large window -> Bigger time uncertainty, smaller frequency uncertainty. dt*df>=C
    # window_length = 32768
    # hop_size = 1024

    # overlap is the shared sample points between windows, artificially raising our data resolution
    overlap = window_size - hop_size

    frequency, time, z = signal.stft(
        data, fs=rate, nperseg=window_size, noverlap=overlap
    )

    # Discard complex numbers, keep only the magnitude
    magnitude = np.abs(z)

    settings = {
        "data_rate_hz": rate,
        "window_length": window_size,
        "hop_size": hop_size,
        "overlap": overlap,
        "window_type": "SciPy's default STFT window",
        "time_bins": len(time),
        "freq_bins": len(frequency),
    }

    with open(f"{output_dir}/settings.txt", "w") as f:
        json.dump(settings, f, indent=4)

    np.savez_compressed(
        f"{output_dir}/data.npz", frequencies=frequency, times=time, magnitude=magnitude
    )
