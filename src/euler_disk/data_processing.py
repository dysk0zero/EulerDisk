# ./src/euler_disk/data_processing.py
import json

import numpy as np
from scipy import signal
from scipy.io import wavfile


def short_time_ft(
    input_path: str, output_dir: str, window_size: int, hop_size: int
) -> None:
    '''
    Short-time Fourier transform of a .wav file.
    Inputs:
        input_path: path to the .wav file
        output_dir: directory to save the output files
        window_size: size of the window for the STFT
        hop_size: number of samples to move the window at each step
    
    Outputs:
        Saves the STFT magnitude, frequencies, and times to a .npz file in the output_dir.
        Saves the settings used for the STFT to a settings.txt file in the output_dir.
    '''
    rate, data = wavfile.read(input_path)

    if len(data.shape) > 1: 
        data = data.mean(axis=1) # Convert to mono by averaging channels

    SFT = signal.ShortTimeFFT(
        win=signal.windows.hann(M=window_size),
        fs=rate,
        hop=hop_size,
        scale_to='magnitude'
    )
    
    z = SFT.stft(data)

    frequency = SFT.f
    time = SFT.t(len(data))

    magnitude = np.abs(z)  # Absolute value of complex spectogram

    overlap = window_size - hop_size

    settings = {
        "data_rate_hz": rate,
        "window_length": window_size,
        "hop_size": hop_size,
        "overlap": overlap,
        "window_type": "Hann",
        "time_bins": len(time),
        "freq_bins": len(frequency),
        "delta_t": SFT.delta_t,
        "delta_f": SFT.delta_f,
    }

    with open(f"{output_dir}/settings.txt", "w") as f:
        json.dump(settings, f, indent=4)

    np.savez_compressed(
        f"{output_dir}/data.npz", frequencies=frequency, times=time, magnitude=magnitude
    )
