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
    fmin = 20
    fmax = 24000
    
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

def load_audio(input_path: str, normalize: bool = False) -> tuple[int, np.ndarray]:
    '''
    Load a .wav file and preprocess it for spectral analysis.
    
    Inputs:
        input_path: path to the .wav file
        normalize: whether to normalize integer PCM audio to roughly [-1, 1]
        
    Returns:
        rate: sampling frequency in Hz
        data: audio data as float64 numpy array (mono)
    '''
    rate, data = wavfile.read(input_path)
    original_dtype = data.dtype

    # Handle stereo audio by converting to mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Ensure data is float for processing
    data = data.astype(np.float64)
    if normalize and np.issubdtype(original_dtype, np.integer):
        data = data / np.iinfo(original_dtype).max

    return rate, data

def _extend_signal(
    data: NDArray[np.float64],
    rate: int,
    extend_to_s: float | None,
    noise_std: float,
    random_seed: int | None,
) -> NDArray[np.float64]:
    if extend_to_s is None:
        extended = data.copy()
    else:
        target_samples = int(np.ceil(extend_to_s * rate))
        if target_samples <= len(data):
            extended = data.copy()
        else:
            extended = np.zeros(target_samples, dtype=np.float64)
            extended[: len(data)] = data

    if noise_std > 0:
        rng = np.random.default_rng(random_seed)
        extended = extended + rng.normal(0.0, noise_std, size=len(extended))

    return extended

def carrier_envelope(
    input_path: str,
    output_dir: str,
    low_cutoff_hz: float,
    high_cutoff_hz: float,
    fir_order: int = 400,
    extend_to_s: float | None = None,
    extension_noise_std: float = 0.0,
    random_seed: int | None = None,
    normalize_audio: bool = True,
    save_carrier_outputs: bool = False,
) -> dict:
    """
    Demodulate an acoustic carrier band with an FIR quadrature filter bank.

    This follows the MATLAB analysis structure:
        B = fir1(order, [low high] / Nyquist)
        Bo = imag(hilbert(B))
        envelope = abs(filter(B, x) + 1j * filter(Bo, x))

    Inputs:
        input_path: path to the .wav file
        output_dir: directory to save output files
        low_cutoff_hz: low edge of the acoustic carrier band
        high_cutoff_hz: high edge of the acoustic carrier band
        fir_order: FIR order; the number of taps is fir_order + 1
        extend_to_s: optional duration to zero-pad the signal to before
            filtering, used to reduce end artifacts near the singularity
        extension_noise_std: optional additive white Gaussian noise standard
            deviation, in the same units as the loaded audio
        random_seed: optional seed for the added noise
        normalize_audio: whether to normalize integer PCM audio to roughly
            [-1, 1], matching MATLAB audioread behavior
        save_carrier_outputs: whether to save the in-phase and quadrature
            carrier outputs in envelope.npz

    Outputs:
        Saves envelope.npz and settings_envelope.txt in output_dir.

    Returns:
        dict with time, envelope, carrier outputs, and sample rate.
    """
    rate, raw = load_audio(input_path, normalize=normalize_audio)
    nyquist = rate / 2
    low = float(low_cutoff_hz)
    high = float(high_cutoff_hz)

    if low <= 0:
        raise ValueError("low_cutoff_hz must be positive.")
    if high <= low:
        raise ValueError("high_cutoff_hz must be greater than low_cutoff_hz.")
    if high >= nyquist:
        raise ValueError("high_cutoff_hz must be below the Nyquist frequency.")
    if fir_order < 2:
        raise ValueError("fir_order must be at least 2.")

    raw_extended = _extend_signal(
        raw,
        rate,
        extend_to_s=extend_to_s,
        noise_std=extension_noise_std,
        random_seed=random_seed,
    )

    bandpass_kernel = signal.firwin(
        numtaps=fir_order + 1,
        cutoff=[low, high],
        pass_zero=False,
        fs=rate,
        window="hamming",
    )
    quadrature_kernel = np.imag(signal.hilbert(bandpass_kernel))

    carrier_in_phase = signal.lfilter(bandpass_kernel, [1.0], raw_extended)
    carrier_quadrature = signal.lfilter(quadrature_kernel, [1.0], raw_extended)
    envelope = np.abs(carrier_in_phase + 1j * carrier_quadrature)
    time = np.arange(len(envelope)) / rate
    center_freq_hz = (low + high) / 2

    settings = {
        "sample_rate_hz": rate,
        "low_cutoff_hz": low,
        "high_cutoff_hz": high,
        "center_freq_hz": center_freq_hz,
        "fir_order": fir_order,
        "num_taps": fir_order + 1,
        "filter_type": "fir_hamming_bandpass_quadrature",
        "input_samples": len(raw),
        "output_samples": len(raw_extended),
        "input_duration_s": len(raw) / rate,
        "output_duration_s": time[-1] if len(time) else 0.0,
        "extend_to_s": extend_to_s,
        "extension_noise_std": extension_noise_std,
        "random_seed": random_seed,
        "normalize_audio": normalize_audio,
        "save_carrier_outputs": save_carrier_outputs,
    }
    with open(f"{output_dir}/settings_envelope.txt", "w") as f:
        json.dump(settings, f, indent=4)

    save_data: dict[str, Any] = {
        "time": time,
        "envelope": envelope,
        "rate": rate,
        "center_freq_hz": center_freq_hz,
        "low_cutoff_hz": low,
        "high_cutoff_hz": high,
    }
    if save_carrier_outputs:
        save_data["carrier_in_phase"] = carrier_in_phase
        save_data["carrier_quadrature"] = carrier_quadrature

    np.savez_compressed(
        f"{output_dir}/envelope.npz",
        **save_data,
    )

    return {
        "time": time,
        "envelope": envelope,
        "rate": rate,
        "center_freq_hz": center_freq_hz,
        "low_cutoff_hz": low,
        "high_cutoff_hz": high,
        "carrier_in_phase": carrier_in_phase,
        "carrier_quadrature": carrier_quadrature,
    }

def preprocess_envelope(
    envelope_path: str,
    output_dir: str,
    downsample_factor: int = 100,
    hp_cutoff_hz: float = 3.0,
    hp_order: int = 2,
) -> dict:
    """
    MATLAB equivalent:

        a1 = resample(a0,1,100)
        [Bhp,Ahp] = butter(2,3/(fs/100/2),'high');
        a2 = filter(Bhp,Ahp,a1);

    Parameters
    ----------
    envelope_path
        Path to envelope.npz produced by carrier_envelope().
    output_dir
        Directory where processed envelope is saved.
    """

    data = np.load(envelope_path)

    envelope = data["envelope"]
    fs = float(data["rate"])

    envelope_ds = signal.resample_poly(
        envelope,
        up=1,
        down=downsample_factor,
    )

    fs_ds = fs / downsample_factor

    sos = signal.butter(
        hp_order,
        hp_cutoff_hz,
        btype="highpass",
        fs=fs_ds,
        output="sos",
    )

    envelope_hp = signal.sosfilt(sos, envelope_ds)

    time = np.arange(len(envelope_hp)) / fs_ds

    settings = {
        "original_rate_hz": fs,
        "downsample_factor": downsample_factor,
        "processed_rate_hz": fs_ds,
        "highpass_order": hp_order,
        "highpass_cutoff_hz": hp_cutoff_hz,
    }

    with open(f"{output_dir}/settings_envelope_preprocessed.txt", "w") as f:
        json.dump(settings, f, indent=4)

    np.savez_compressed(
        f"{output_dir}/envelope_preprocessed.npz",
        time=time,
        envelope=envelope_hp,
        rate=fs_ds,
    )

    return {
        "time": time,
        "envelope": envelope_hp,
        "rate": fs_ds,
    }

def envelope_spectrogram(
    envelope_path: str,
    output_dir: str,
    window_length: int = 512,
    overlap: int = 500,
    nfft: int = 4096,
) -> dict:
    """
    MATLAB equivalent:

        [A0,f,ts] =
            spectrogram(a2,hamming(512),500,4096,fs)

    Returns the COMPLEX STFT.
    """

    data = np.load(envelope_path)

    envelope = data["envelope"]
    fs = float(data["rate"])

    frequency, time, spectrum = signal.spectrogram(
        envelope,
        fs=fs,
        window="hamming",
        nperseg=window_length,
        noverlap=overlap,
        nfft=nfft,
        detrend=False,
        scaling="spectrum",
        mode="complex",
    )

    np.savez_compressed(
        f"{output_dir}/envelope_spectrogram.npz",
        frequency=frequency,
        time=time,
        spectrum=spectrum,
    )

    settings = {
        "window": "Hamming",
        "window_length": window_length,
        "overlap": overlap,
        "nfft": nfft,
        "sample_rate_hz": fs,
    }

    with open(f"{output_dir}/settings_spectrogram.txt", "w") as f:
        json.dump(settings, f, indent=4)

    return {
        "frequency": frequency,
        "time": time,
        "spectrum": spectrum,
    }

def peak_interpolation(
    psd: np.ndarray,
    frequency: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Reproduce MATLAB's three-bin interpolation exactly.
    """

    refined = frequency[indices].copy()

    df = frequency[1] - frequency[0]

    for k in range(len(indices)):

        i = indices[k]

        if i == 0:
            continue

        if i == len(frequency) - 1:
            continue

        peak = psd[i, k]

        ya = psd[i - 1, k] - peak
        yb = psd[i + 1, k] - peak

        denom = ya + yb

        if denom == 0:
            continue

        dx = 0.5 * (ya - yb) / denom

        refined[k] += dx * df

    return refined

def extract_precession_ridge(
    spectrogram_path: str,
    output_dir: str,
    frequency_min_hz: float = 20,
    frequency_max_hz: float = 60,
) -> dict:
    """
    Extract dominant ridge exactly like MATLAB.
    """

    data = np.load(spectrogram_path)

    frequency = data["frequency"]
    time = data["time"]
    spectrum = data["spectrum"]

    psd = np.abs(spectrum) ** 2

    mask = (
        (frequency >= frequency_min_hz)
        & (frequency <= frequency_max_hz)
    )

    f = frequency[mask]

    psd = psd[mask]

    peak_indices = np.argmax(psd, axis=0)

    ridge_raw = f[peak_indices]

    ridge_interp = peak_interpolation(
        psd,
        f,
        peak_indices,
    )

    np.savez_compressed(
        f"{output_dir}/precession_ridge.npz",
        time=time,
        frequency=ridge_interp,
        frequency_raw=ridge_raw,
        envelope_frequency_min_hz=frequency_min_hz,
        envelope_frequency_max_hz=frequency_max_hz,
    )

    return {
        "time": time,
        "frequency": ridge_interp,
        "frequency_raw": ridge_raw,
        "spectrogram": psd,
        "frequency_axis": f,
    }

def fit_precession_power_law(
    ridge_path: str,
    output_dir: str,
    t0_initial: float = 26.08,
) -> dict:
    """
    Initial power-law fit of the precession frequency.
    """

    data = np.load(ridge_path)

    time = data["time"]
    frequency = data["frequency"]

    x = np.log10(t0_initial - time)

    mask = (
        (time < t0_initial)
        & (x > -0.5)
        & (x < 1.0)
    )

    time = time[mask]
    frequency = frequency[mask]
    x = x[mask]
    y = np.log10(frequency)

    slope, intercept = np.polyfit(x, y, 1)

    alpha = -slope
    k = 10**intercept

    y_model = intercept + slope * x
    frequency_model = k * (1.0 / (t0_initial - time)) ** alpha

    np.savez_compressed(
        f"{output_dir}/precession_powerlaw_fit.npz",
        time=time,
        frequency=frequency,
        log_time=x,
        log_frequency=y,
        log_frequency_model=y_model,
        frequency_model=frequency_model,
        alpha=alpha,
        k=k,
        t0=t0_initial,
    )

    return {
        "time": time,
        "frequency": frequency,
        "log_time": x,
        "log_frequency": y,
        "log_frequency_model": y_model,
        "frequency_model": frequency_model,
        "alpha": alpha,
        "k": k,
        "t0": t0_initial,
    }

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

# LEGACY FUNCTIONS

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
    with open(f"{output_dir}/settings_hilbert_envelope.txt", "w") as f:
        json.dump(settings, f, indent=4)

    np.savez_compressed(
        f"{output_dir}/hilbert_envelope.npz",
        time=time,
        envelope=envelope,
        rate=rate,
        center_freq_hz=f_peak,
    )

    return {"time": time, "envelope": envelope, "center_freq_hz": f_peak}
