# ./src/euler_disk/data_processing.py
import json

import numpy as np
from scipy import signal
from scipy.io import wavfile


def short_time_ft(input_path: str, output_dir: str) -> None:
    rate, data = wavfile.read(input_path)

    # Large window -> Bigger time uncertainty, smaller frequency uncertainty. dt*df>=C
    window_length = 1024
    hop_size = 128

    # overlap is the shared sample points between windows, artificially raising our data resolution
    overlap = window_length - hop_size

    frequency, time, z = signal.stft(
        data, fs=rate, nperseg=window_length, noverlap=overlap
    )

    # Discard complex numbers, keep only the magnitude
    magnitude = np.abs(z)

    settings = {
        "data_rate_hz": rate,
        "window_length": window_length,
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
