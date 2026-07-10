# ./src/euler_disk/data_processing.py
import json

import numpy as np
from scipy import signal
from scipy.io import wavfile

def welch(
    input_path: str, output_dir: str, window_size: int, hop_size: int
    ) -> None:
    '''
    Uses Welch's method to estimate the power spectral density of a signal.
    Inputs:
        input_path: path to the .wav file
        output_dir: directory to save the output files
        window_size: length of each segment (nperseg)
        hop_size: number of points to advance between segments
    
    Outputs:
        Saves the PSD frequencies and power values to a .npz file in the output_dir.
        Saves the settings used for the PSD to a settings.txt file in the output_dir.
        '''
    rate, data = load_audio(input_path)

    noverlap=(window_size - hop_size)

    frequency, power = signal.welch(
        data,
        fs=rate,
        window='hann',  # Periodic Hann window
        nperseg=window_size,
        noverlap=noverlap,
        scaling='density'  # Units: V²/Hz; use 'spectrum' for V²
    )
    
    settings = {
        "data_rate_hz": rate,
        "window_length": window_size,
        "hop_size": hop_size,
        "overlap": noverlap,
        "window_type": "Hann",
        "freq_bins": len(frequency),
        "scaling": "density",  # or 'spectrum'
    }

    with open(f"{output_dir}/settings_welch.txt", "w") as f:
        json.dump(settings, f, indent=4)

    np.savez_compressed(
        f"{output_dir}/data_welch.npz", frequencies=frequency, power=power
    )


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
    rate, data = load_audio(input_path)

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

    with open(f"{output_dir}/settings_stft.txt", "w") as f:
        json.dump(settings, f, indent=4)

    np.savez_compressed(
        f"{output_dir}/data.npz", frequencies=frequency, times=time, magnitude=magnitude
    )

def load_audio(input_path: str) -> tuple[int, np.ndarray]:
    '''
    Load a .wav file and preprocess it for spectral analysis.
    
    Inputs:
        input_path: path to the .wav file
        
    Returns:
        rate: sampling frequency in Hz
        data: audio data as float64 numpy array (mono)
    '''
    rate, data = wavfile.read(input_path)

    # Handle stereo audio by converting to mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Ensure data is float for processing
    data = data.astype(np.float64)

    return rate, data