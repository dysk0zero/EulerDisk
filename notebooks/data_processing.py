# Generated from: data_processing.ipynb
# Converted at: 2026-07-13T13:15:20.071Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os
from time import time

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
# We must find the overall frequency contents of our signal and identify strong acoustic carrier bands for later demodulation.
# First we perform an stimation of the power spectral density, computing the overall frequency contents of our signal over the recorded time.


window_size=4096
hop_size=2048

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

# ## Step 2: Identify Carrier Peaks
# We use our own function that uses scipy.find_peaks to find strong acoustic carriers and then choose our bandpass for AM processing.


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

# Select bandpass for Step 3.
carrier_low_hz = 4600
carrier_high_hz = 5400

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_yscale("linear")
ax.plot(data_welch["frequencies"], welch_db, label="Power Spectral Density", color="tab:blue")
ax.scatter(peak_data["frequencies"], peak_db, color="tab:red", s=50, marker="v", label=f"Peaks (n={len(peak_data['frequencies'])})", zorder=5)
ax.axvspan(
    carrier_low_hz,
    carrier_high_hz,
    alpha=0.2,
    color="tab:green",
    label="Selected carrier band",
)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power Spectral Density (dB)")
ax.set_title("Power Spectrum with Detected Peaks - Welch's Method")
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(f_min, 20000)
ax.set_ylim(-100)
ax.legend()

fig.tight_layout()
plt.show()

# # Step 3: Carrier-Band Quadrature Demodulation
# We isolate the selected acoustic carrier band with an FIR filter bank and extract its envelope.


carrier_fir_order = 400
extend_to_s = 30.0
extension_noise_std = 1e-3
normalize_audio = True

ed.carrier_envelope(
    input_path=input_dir,
    output_dir=output_dir,
    low_cutoff_hz=carrier_low_hz,
    high_cutoff_hz=carrier_high_hz,
    fir_order=carrier_fir_order,
    extend_to_s=extend_to_s,
    extension_noise_std=extension_noise_std,
    random_seed=0,
    normalize_audio=normalize_audio,
    save_carrier_outputs=False,
)

# ## Step 4: Envelope
# The envelope is the magnitude of the in-phase and quadrature carrier-band outputs.

envelope_data = np.load(f"{output_dir}/envelope.npz")

envelope_time = envelope_data["time"]
envelope = envelope_data["envelope"]

fig, envelope_axes = plt.subplots(1, 2, figsize=(20, 4), squeeze=False)
envelope_ax = envelope_axes[0, 0]
envelope_zoom_ax = envelope_axes[0, 1]

envelope_ax.plot(envelope_time, envelope, color="g")
envelope_ax.set_xlabel("Time (s)")
envelope_ax.set_ylabel("Envelope Amplitude")
envelope_ax.set_title("Envelope of the Filtered Signal")
envelope_ax.grid(True, which="both", alpha=0.3)
envelope_ax.set_xlim(envelope_time[0], envelope_time[-1])

envelope_zoom_ax.plot(envelope_time, envelope, color="g")
envelope_zoom_ax.set_xlabel("Time (s)")
envelope_zoom_ax.set_ylabel("Envelope Amplitude")
envelope_zoom_ax.set_title("Zoomed Envelope of the Filtered Signal - Last 4 seconds")
envelope_zoom_ax.grid(True, which="both", alpha=0.3)
envelope_zoom_ax.set_xlim(24.0, 26.1)
for ax in (envelope_ax, envelope_zoom_ax):
    ax.axvline(
        26.05,
        color="tab:red",
        linestyle="--",
        alpha=0.5,
        label="Approx. end of motion",
    )

fig.tight_layout()
plt.show()

# ## Step 5: Envelope Processing



# The physical precession fundamental is expected in the low-frequency amplitude modulation, not in the acoustic carrier band itself.

rate = float(envelope_data["rate"])
carrier_center_hz = float(envelope_data["center_freq_hz"])

spectrogram_start = 23.0
spectrogram_end = min(26.1, envelope_time[-1])
start_sample = int(spectrogram_start * rate)
end_sample = int(spectrogram_end * rate)

envelope_segment = envelope[start_sample:end_sample]
envelope_segment = signal.detrend(envelope_segment, type="constant")

envelope_spectrogram_nperseg = min(32768, len(envelope_segment))
envelope_spectrogram_noverlap = min(28672, envelope_spectrogram_nperseg - 1)

(
    envelope_frequencies,
    envelope_spectrogram_times,
    envelope_spectrogram_power,
) = signal.spectrogram(
    envelope_segment,
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
fundamental_max_hz = 80
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

np.savez_compressed(
    f"{output_dir}/precession_ridge.npz",
    time=envelope_spectrogram_times,
    frequency=fundamental_ridge,
    carrier_center_hz=carrier_center_hz,
    frequency_min_hz=fundamental_min_hz,
    frequency_max_hz=fundamental_max_hz,
    time_start_s=spectrogram_start,
    time_end_s=spectrogram_end,
)

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
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Demodulated Envelope Spectrogram - Fundamental Band")
ax.set_xlim(spectrogram_start, spectrogram_end)
ax.set_ylim(fundamental_min_hz, fundamental_max_hz)
ax.grid(True, which="both", alpha=0.2)
ax.legend(loc="upper left")

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Envelope Power Spectral Density (dB/Hz)")

fig.tight_layout()
plt.show()
