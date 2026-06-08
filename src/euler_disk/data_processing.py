# ./src/euler_disk/data_processing.py
import numpy as np
from scipy import signal
from scipy.io import wavfile


def short_time_ft(input_path: str, output_path: str) -> None:
    rate, data = wavfile.read(input_path)

    # Large window -> Bigger time uncertainty, smaller frequency uncertainty. dt*df>=C
    window_length = 1024
    hop_size = 128

    # overlap = window_lenght - hop_lenght
    # overlap is the shared sample points between windows, artificially raising our data resolution
    overlap = window_length - hop_size

    frequency, time, z = signal.stft(
        data, fs=rate, nperseg=window_length, noverlap=overlap
    )

    settings = {
        "data_rate_hz": rate,
        "window_length": window_length,
        "overlap": overlap,
        "window_type": "hann",  # SciPy's default STFT window
    }

    np.savez_compressed(
        output_path,
        frequencies=frequencies,
        times=times,
        stft_matrix=Zxx,
        settings=settings,
    )
