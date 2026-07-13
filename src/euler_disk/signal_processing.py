# ./src/euler_disk/data_processing.py
"""
Data processing module for spectral analysis.

Functions:
    load_audio: Load and preprocess .wav files
    welch: Compute power spectral density using Welch's method
    short_time_ft: Compute STFT using ShortTimeFFT
    find_peaks_psd: Find peaks in PSD data
"""

import json
from typing import Any, TypeAlias, TypedDict, NotRequired, cast

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.io import wavfile

PeakCriterion: TypeAlias = float | tuple[float, float] | None

class PeakResults(TypedDict):
    peak_indices: NDArray[np.intp]
    frequencies: NDArray[np.float64]
    powers: NDArray[np.float64]
    heights: NotRequired[NDArray[np.float64]]
    prominences: NotRequired[NDArray[np.float64]]
    widths: NotRequired[NDArray[np.float64]]
    left_ips: NotRequired[NDArray[np.float64]]
    right_ips: NotRequired[NDArray[np.float64]]

fmin = 20
fmax = 24000

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
    
    band = (frequency >= fmin) & (frequency <= fmax)
    frequency = frequency[band]
    power = power[band]

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

def peaks(
    input_path: str,
    output_dir: str,
    height: PeakCriterion = None,
    threshold: PeakCriterion = None,
    distance: int | None = None,
    prominence: PeakCriterion = None,
    width: PeakCriterion = None,
) -> PeakResults:
    """
    Find peaks in power spectral density data.
    
    Inputs:
        psd_path: path to the .npz file from welch() output
        output_dir: directory to save the output files
        height: required height of peaks (same units as power)
        threshold: required threshold (vertical distance to neighbors)
        distance: minimal horizontal distance between peaks (in samples)
        prominence: required prominence of peaks
        width: required width of peaks (in samples)
    
    Outputs:
        Saves peak data to peak_data.npz in output_dir.
        Saves settings to settings_peaks.txt in output_dir.
        
    Returns:
        dict with peak indices, frequencies, powers, and properties
    """
    data = np.load(input_path)
    frequency = data["frequencies"]
    power = data["power"]

    power_db = 10 * np.log10(power + 1e-20)

    peak_kwargs: dict[str, Any] = {
        "height": height, "threshold": threshold, "distance": distance,
        "prominence": prominence, "width": width,
    }
    
    peaks, _props = signal.find_peaks(power_db, **peak_kwargs)
    properties: dict[str, Any] = dict(_props)
    
    peak_freqs = frequency[peaks]
    peak_powers = power[peaks]

    results: PeakResults = {
        "peak_indices": peaks,
        "frequencies": peak_freqs,
        "powers": peak_powers,
    }
    if "peak_heights" in properties:
        results["heights"] = properties["peak_heights"]
    if "prominences" in properties:
        results["prominences"] = properties["prominences"]
    if "widths" in properties:
        results["widths"] = properties["widths"]
        results["left_ips"] = properties["left_ips"]
        results["right_ips"] = properties["right_ips"]

    settings = {
        "num_peaks": len(peaks),
        "height": height,
        "threshold": threshold,
        "distance": distance,
        "prominence": prominence,
        "width": width,
    }

    with open(f"{output_dir}/settings_peaks.txt", "w") as f:
        json.dump(settings, f, indent=4)

    saveable: dict[str, NDArray[Any]] = {
        k: cast("NDArray[Any]", v)
        for k, v in results.items()
        if k in (
            "peak_indices", "frequencies", "powers",
            "heights", "prominences", "widths",
            "left_ips", "right_ips",
        )
    }
    np.savez_compressed(
        f"{output_dir}/peak_data.npz",
        allow_pickle=False,
        **saveable,
    )

    return results

def bandpass(
    input_path: str,
    output_dir: str,
    center_freq_hz: float,
    bandwidth_hz: float = 50.0,
    filter_order: int = 4,
) -> dict:
    """
    Isolate a selected frequency band via zero-phase Butterworth bandpass.

    Inputs:
        input_path: path to the .wav file (raw audio)
        output_dir: directory to save the output files
        center_freq_hz: center frequency of the selected bandpass
        bandwidth_hz: total bandpass width in Hz
        filter_order: Butterworth filter order (per stage; filtfilt doubles
            the effective rolloff)
    Outputs:
        Saves filtered signal to filtered_signal.npz in output_dir.
        Saves settings to settings_bandpass.txt in output_dir.
    Returns:
        dict with the filtered signal, sample rate, and center frequency
    """
    rate, raw = load_audio(input_path)

    f_center = float(center_freq_hz)
    if f_center <= 0:
        raise ValueError("center_freq_hz must be positive.")
    if f_center >= rate / 2:
        raise ValueError("center_freq_hz must be below the Nyquist frequency.")

    low = max(f_center - bandwidth_hz / 2, 0.1)   # keep strictly > 0
    high = min(f_center + bandwidth_hz / 2, rate / 2 * 0.999)  # stay under Nyquist

    sos = signal.butter(
        filter_order, [low, high], btype="bandpass", fs=rate, output="sos"
    )
    filtered = signal.sosfiltfilt(sos, raw)

    settings = {
        "center_freq_hz": f_center,
        "bandwidth_hz": bandwidth_hz,
        "low_cutoff_hz": low,
        "high_cutoff_hz": high,
        "filter_order": filter_order,
        "filter_type": "butterworth_bandpass_zero_phase",
        "sample_rate_hz": rate,
    }
    with open(f"{output_dir}/settings_bandpass.txt", "w") as f:
        json.dump(settings, f, indent=4)

    np.savez_compressed(
        f"{output_dir}/filtered_signal.npz",
        signal=filtered,
        rate=rate,
        center_freq_hz=f_center,
    )

    return {
        "signal": filtered,
        "rate": rate,
        "center_freq_hz": f_center,
    }

def envelope(
    filtered_signal_path: str,
    output_dir: str,
    smooth_window: int | None = None,
) -> dict:
    """
    Extract the amplitude envelope of a filtered signal via the Hilbert transform.

    Inputs:
        filtered_signal_path: path to filtered_signal.npz from bandpass_isolate
        output_dir: directory to save the output files
        smooth_window: optional moving-average window (in samples) to
            smooth the envelope; None skips smoothing
    Outputs:
        Saves envelope, time axis, and analytic signal to envelope.npz.
        Saves settings to settings_envelope.txt.
    Returns:
        dict with time, envelope, and center frequency
    """
    data = np.load(filtered_signal_path)
    filtered = data["signal"]
    rate = float(data["rate"])
    f_peak = float(data["center_freq_hz"])

    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)

    if smooth_window is not None and smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        envelope = np.convolve(envelope, kernel, mode="same")

    time = np.arange(len(filtered)) / rate

    settings = {
        "sample_rate_hz": rate,
        "center_freq_hz": f_peak,
        "smooth_window": smooth_window,
        "n_samples": len(filtered),
        "duration_s": time[-1] if len(time) else 0.0,
    }
    with open(f"{output_dir}/settings_envelope.txt", "w") as f:
        json.dump(settings, f, indent=4)

    np.savez_compressed(
        f"{output_dir}/envelope.npz",
        time=time,
        envelope=envelope,
        rate=rate,
        center_freq_hz=f_peak,
    )

    return {"time": time, "envelope": envelope, "center_freq_hz": f_peak}
