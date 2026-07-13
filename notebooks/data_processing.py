# Generated from: data_processing.ipynb
# Converted at: 2026-07-13T13:15:20.071Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os

import numpy as np
import euler_disk as ed
import matplotlib.pyplot as plt
from scipy import signal

# # Notebook Data Analysis of Euler Disk
# Note: This is the single file version of the data analysis notebook, intended to perform analysis of all of the experimental data in every run.


# ## Input Data Folder
# Under this folder are all the wav files.


input_dir = "../data/wav/recording_04.wav"
output_dir = "../data/output/recording_04/"
os.makedirs(output_dir, exist_ok=True)

# ## Step 1: Welch's Method
# We must find the overall frequency contents of our signal and find the harmonics of the fundamental frequncy to then later do a correct bandpass filtering for further processing.
# First we perform an stimation of the power spectral density, computing the overall frequency contents of our signal over the recorded time.


window_size=4096
hop_size=1024

ed.welch(input_path=input_dir, output_dir=output_dir, window_size=window_size, hop_size=hop_size)

data_welch = np.load(f"{output_dir}/data_welch.npz")
freq, power = data_welch["frequencies"], data_welch["power"]

f_min, f_max = 20, 24000

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(freq, power)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power Spectral Density (V²/Hz)")
ax.set_title("Welch's Method - Power Spectrum")
ax.set_xlim(f_min, f_max)
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
plt.show()

# ## Step 2: Identify Harmonic's Peaks
# We use our own function that uses scipy.find_peaks to find the exact first harmonics and then choose our bandpass for AM processing.


peaks = ed.peaks(
    input_path=f"{output_dir}/data_welch.npz",
    output_dir=f"{output_dir}",
    prominence=6,   # dB
    distance=2,     # bins (~47 Hz at Δf≈23.4 Hz)
)

n = len(peaks["frequencies"])
print(f"\nFound {n} peaks:\n")
print(f"{'Frequency (Hz)':<18} {'Power (V²/Hz)':<18} {'Prominence (dB)':<18}")
print("-" * 54)

freqs = peaks["frequencies"]
powers = peaks["powers"]
proms = peaks.get("prominences", [None] * n)

for f, p, pr in zip(freqs, powers, proms):
    prom_str = f"{pr:<18.2f}" if pr is not None else f"{'—':<18}"
    print(f"{f:<18.2f} {p:<18.3e} {prom_str}")

peak_data = np.load(f"{output_dir}/peak_data.npz")

welch_db = 10 * np.log10(data_welch["power"] + 1e-20)
peak_db = 10 * np.log10(peak_data["powers"] + 1e-20)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_yscale("linear")
ax.plot(data_welch["frequencies"], welch_db, label="Power Spectral Density", color="tab:blue")
ax.scatter(peak_data["frequencies"], peak_db, color="tab:red", s=50, marker="v", label=f"Peaks (n={len(peak_data['frequencies'])})", zorder=5)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power Spectral Density (dB)")
ax.set_title("Power Spectrum with Detected Peaks - Welch's Method")
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(f_min, 20000)
ax.set_ylim(-100)
ax.legend()
fig.tight_layout()
plt.show()

# # Step 3: Bandpass Filtering
# We then apply a bandpass filter to isolate the selected acoustic carrier band.


carrier_center_hz = 5039.06

ed.bandpass(
    input_path=input_dir,
    output_dir=output_dir,
    center_freq_hz=carrier_center_hz,
    bandwidth_hz=250,
    filter_order=4,
)

# ## Step 4: Envelope 
# Now we must use amplitude demodulation to reduce the AM artifacts in our signal. We perform a Hilbert transform to obtain the *analytic signal*, a complex value whose magnitude is the amplitude, obtaining the envelope.


ed.envelope(
    filtered_signal_path=f"{output_dir}/filtered_signal.npz",
    output_dir=output_dir,
    smooth_window=1920,
)

envelope_data = np.load(f"{output_dir}/envelope.npz")

time, envelope = envelope_data["time"], envelope_data["envelope"]

fig, envelope_axes = plt.subplots(1, 2, figsize=(20, 4), squeeze=False)
envelope_ax = envelope_axes[0, 0]
envelope_zoom_ax = envelope_axes[0, 1]

envelope_ax.plot(time, envelope, color="g")
envelope_ax.set_xlabel("Time (s)")
envelope_ax.set_ylabel("Envelope Amplitude")
envelope_ax.set_title("Envelope of the Filtered Signal")
envelope_ax.grid(True, which="both", alpha=0.3)
envelope_ax.set_xlim(time[0], time[-1])

envelope_zoom_ax.plot(time, envelope, color="g")
envelope_zoom_ax.set_xlabel("Time (s)")
envelope_zoom_ax.set_ylabel("Envelope Amplitude")
envelope_zoom_ax.set_title("Zoomed Envelope of the Filtered Signal - Last 4 seconds")
envelope_zoom_ax.grid(True, which="both", alpha=0.3)
envelope_zoom_ax.set_xlim(22, 26.1)

fig.tight_layout()
plt.show()

# ## Step 5: Spectrogram
# Now that we applied our AM correction filters and narrow our search to a harmonic band, we plot the spectrogram of the processed signal near the singularity.

filtered_data = np.load(f"{output_dir}/filtered_signal.npz")
processed_signal = filtered_data["signal"]
rate = float(filtered_data["rate"])
center_freq = float(filtered_data["center_freq_hz"])

singularity_time = len(processed_signal) / rate
spectrogram_window_s = 4.0
spectrogram_start = max(0.0, singularity_time - spectrogram_window_s)

start_sample = int(spectrogram_start * rate)
processed_tail = processed_signal[start_sample:]

spectrogram_nperseg = min(2048, len(processed_tail))
spectrogram_noverlap = min(1792, spectrogram_nperseg - 1)

frequencies, spectrogram_times, spectrogram_power = signal.spectrogram(
    processed_tail,
    fs=rate,
    window="hann",
    nperseg=spectrogram_nperseg,
    noverlap=spectrogram_noverlap,
    detrend="constant",
    scaling="density",
    mode="psd",
)
spectrogram_times = spectrogram_times + spectrogram_start
spectrogram_db = 10 * np.log10(spectrogram_power + 1e-20)

spectrogram_bandwidth_hz = 350
freq_mask = (
    (frequencies >= center_freq - spectrogram_bandwidth_hz / 2)
    & (frequencies <= center_freq + spectrogram_bandwidth_hz / 2)
)

band_frequencies = frequencies[freq_mask]
band_spectrogram_db = spectrogram_db[freq_mask, :]
if len(band_frequencies) == 0:
    raise ValueError("No spectrogram frequency bins found around the processed band.")
ridge_frequency = band_frequencies[np.argmax(band_spectrogram_db, axis=0)]

fig, ax = plt.subplots(figsize=(10, 5))
mesh = ax.pcolormesh(
    spectrogram_times,
    band_frequencies,
    band_spectrogram_db,
    shading="auto",
    cmap="magma",
)
ax.plot(
    spectrogram_times,
    ridge_frequency,
    color="cyan",
    linewidth=1.2,
    label="Dominant frequency ridge",
)
ax.axvline(singularity_time, color="white", linestyle="--", linewidth=1, alpha=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Spectrogram of Processed Signal - Last 4 seconds")
ax.set_xlim(spectrogram_start, singularity_time)
ax.set_ylim(band_frequencies[0], band_frequencies[-1])
ax.grid(True, which="both", alpha=0.2)
ax.legend(loc="upper left")

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Power Spectral Density (dB/Hz)")

fig.tight_layout()
plt.show()

# ## Step 6: Low-Frequency Spectrogram of the Envelope
# The physical precession fundamental is expected in the low-frequency amplitude modulation, not in the acoustic carrier band itself.

envelope_tail = envelope[start_sample:]
envelope_tail = envelope_tail - np.mean(envelope_tail)

envelope_spectrogram_nperseg = min(32768, len(envelope_tail))
envelope_spectrogram_noverlap = min(28672, envelope_spectrogram_nperseg - 1)

(
    envelope_frequencies,
    envelope_spectrogram_times,
    envelope_spectrogram_power,
) = signal.spectrogram(
    envelope_tail,
    fs=rate,
    window="hann",
    nperseg=envelope_spectrogram_nperseg,
    noverlap=envelope_spectrogram_noverlap,
    detrend="constant",
    scaling="density",
    mode="psd",
)
envelope_spectrogram_times = envelope_spectrogram_times + spectrogram_start
envelope_spectrogram_db = 10 * np.log10(envelope_spectrogram_power + 1e-20)

fundamental_min_hz = 20
fundamental_max_hz = 100
fundamental_mask = (
    (envelope_frequencies >= fundamental_min_hz)
    & (envelope_frequencies <= fundamental_max_hz)
)

fundamental_frequencies = envelope_frequencies[fundamental_mask]
fundamental_spectrogram_db = envelope_spectrogram_db[fundamental_mask, :]
if len(fundamental_frequencies) == 0:
    raise ValueError("No envelope spectrogram bins found in the fundamental band.")
fundamental_ridge = fundamental_frequencies[
    np.argmax(fundamental_spectrogram_db, axis=0)
]

fig, ax = plt.subplots(figsize=(10, 5))
mesh = ax.pcolormesh(
    envelope_spectrogram_times,
    fundamental_frequencies,
    fundamental_spectrogram_db,
    shading="auto",
    cmap="viridis",
)
ax.plot(
    envelope_spectrogram_times,
    fundamental_ridge,
    color="white",
    linewidth=1.2,
    label="Dominant envelope ridge",
)
ax.axvline(singularity_time, color="white", linestyle="--", linewidth=1, alpha=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Envelope Spectrogram - Fundamental Band")
ax.set_xlim(spectrogram_start, singularity_time)
ax.set_ylim(fundamental_min_hz, fundamental_max_hz)
ax.grid(True, which="both", alpha=0.2)
ax.legend(loc="upper left")

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Envelope Power Spectral Density (dB/Hz)")

fig.tight_layout()
plt.show()
