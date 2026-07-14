import os

import numpy as np
import euler_disk as ed
import matplotlib.pyplot as plt

# # Notebook Data Analysis of Euler Disk
# Note: This is the single file version of the data analysis notebook, intended to perform analysis of all of the experimental data in every run.


# ## Input Data Folder
# Under this folder are all the wav files.


input_dir = "../../data/wav/recording_04.wav"
output_dir = "../../data/output/recording_04/"
os.makedirs(output_dir, exist_ok=True)

# ## Step 1: Welch's Method
# We must find the overall frequency contents of our signal and identify strong acoustic carrier bands for later demodulation.
# First we perform an stimation of the power spectral density, computing the overall frequency contents of our signal over the recorded time.


window_size=4096
hop_size=2048

ed.welch(input_path=input_dir, output_dir=output_dir, window_size=window_size, hop_size=hop_size)

data_welch = np.load(f"{output_dir}/data_welch.npz")
freq, power = data_welch["frequencies"], data_welch["power"]

# ## Step 2: Identify Resonant Peaks
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
ax.plot(data_welch["frequencies"], welch_db, label="Power Spectral Density", linewidth=1.5, color="tab:blue")
ax.scatter(peak_data["frequencies"], peak_db, color="tab:red", s=50, marker="v", label=f"Peaks (n={len(peak_data['frequencies'])})", zorder=5)
ax.axvspan(
    carrier_low_hz,
    carrier_high_hz,
    alpha=0.15,
    color="tab:green",
    label="Selected carrier band",
)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power Spectral Density (dB)")
ax.set_title("Power Spectrum and Resonant Peaks - Welch's Method")
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(20, 20000)
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

# ## Step 4: Processed Envelope
# Downsample the carrier envelope and remove its DC component prior to
# spectrogram analysis, following the MATLAB reference pipeline.

downsample_factor = 100
hp_cutoff_hz = 3.0
hp_order = 2

ed.preprocess_envelope(
    envelope_path=f"{output_dir}/envelope.npz",
    output_dir=output_dir,
    downsample_factor=downsample_factor,
    hp_cutoff_hz=hp_cutoff_hz,
    hp_order=hp_order,
)

preprocessed_data = np.load(f"{output_dir}/envelope_preprocessed.npz")

preprocessed_time = preprocessed_data["time"]
preprocessed_envelope = preprocessed_data["envelope"]
preprocessed_rate = float(preprocessed_data["rate"])

fig, axes = plt.subplots(1, 2, figsize=(20, 4), squeeze=False)

ax_full = axes[0, 0]
ax_zoom = axes[0, 1]

ax_full.plot(preprocessed_time, preprocessed_envelope, color="tab:blue")
ax_full.set_xlabel("Time (s)")
ax_full.set_ylabel("Envelope Amplitude")
ax_full.set_title("Processed Envelope at the Beginning")
ax_full.grid(True, which="both", alpha=0.3)
ax_full.set_xlim(4.0, 7.1)

ax_zoom.plot(preprocessed_time, preprocessed_envelope, color="tab:blue")
ax_zoom.set_xlabel("Time (s)")
ax_zoom.set_ylabel("Envelope Amplitude")
ax_zoom.set_title("Processed Envelope at the Singularity")
ax_zoom.grid(True, which="both", alpha=0.3)
ax_zoom.set_xlim(23.0, 26.1)

fig.tight_layout()
plt.show()

# ## Step 5: Spectrogram
# Compute the spectrogram of the downsampled, high-pass filtered envelope
# using the same parameters as the MATLAB reference.

window_length = 512
overlap = 500
nfft = 4096

ed.envelope_spectrogram(
    envelope_path=f"{output_dir}/envelope_preprocessed.npz",
    output_dir=output_dir,
    window_length=window_length,
    overlap=overlap,
    nfft=nfft,
)

spectrogram_data = np.load(
    f"{output_dir}/envelope_spectrogram.npz"
)

spectrogram_frequency = spectrogram_data["frequency"]
spectrogram_time = spectrogram_data["time"]
spectrogram = spectrogram_data["spectrum"]

#
# MATLAB:
# imagesc(10*log10(abs(A0)+0.01))
#
spectrogram_db = 10 * np.log10(np.abs(spectrogram) + 0.01)

frequency_min_hz = 15
frequency_max_hz = 45

frequency_mask = (
    (spectrogram_frequency >= frequency_min_hz)
    & (spectrogram_frequency <= frequency_max_hz)
)

fig, ax = plt.subplots(figsize=(10, 5))

mesh = ax.pcolormesh(
    spectrogram_time,
    spectrogram_frequency[frequency_mask],
    spectrogram_db[frequency_mask],
    shading="auto",
    cmap="plasma",
)

ax.set_xlim(24.0, 26.1)
ax.set_ylim(frequency_min_hz, frequency_max_hz)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Spectrogram at Singularity")

ax.grid(True, which="both", alpha=0.2)

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Magnitude (dB)")


fig.tight_layout()
plt.show()

# ## Step 6: Extract Precession Ridge
# We extract the dominant precession ridge from the spectrogram using the same parameters as the MATLAB reference.
ridge = ed.extract_precession_ridge(
    spectrogram_path=f"{output_dir}/envelope_spectrogram.npz",
    output_dir=output_dir,
    frequency_min_hz=frequency_min_hz,
    frequency_max_hz=frequency_max_hz,
)

fig, ax = plt.subplots(figsize=(10, 5))

mesh = ax.pcolormesh(
    spectrogram_time,
    spectrogram_frequency[frequency_mask],
    spectrogram_db[frequency_mask],
    shading="auto",
    cmap="plasma",
)

plt.plot(
    ridge["time"],
    ridge["frequency"],
    color="white",
    linewidth=2,
    label="Dominant ridge",
    alpha=0.6,
)

ax.set_xlim(24.0, 26.1)
ax.set_ylim(frequency_min_hz, frequency_max_hz)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Dominant Precession Ridge")

ax.grid(True, which="both", alpha=0.2)
ax.legend(loc="upper left")

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Magnitude (dB)")

fig.tight_layout()
plt.show()

# ## Step 7: Linear fit over log-log of the precession ridge
from scipy.stats import linregress

ts = ridge["time"]
fmax_smooth = ridge["frequency"]

lf = np.log10(fmax_smooth)

# Initial guess
t0 = 26.08

x = np.log10(t0 - ts)

cond = ts < t0

result = linregress(x[cond], lf[cond])

alpha = -result.slope
k = 10**result.intercept

print(f"Initial estimate: alpha={alpha:.4f}, k={k:.4f}")

# Restrict fitting range
cond = (x > -0.5) & (x < 1) & (ts < t0)

x_fit = x[cond]
lf_fit = lf[cond]
ts_fit = ts[cond]

result = linregress(x_fit, lf_fit)

alpha = -result.slope
k = 10**result.intercept

f_model = k * (1/(t0-ts_fit))**alpha

best_error = np.inf

for t0_test in np.arange(25, 27.001, 0.05):

    cond = ts_fit < t0_test

    if np.sum(cond) < 3:
        continue

    x_test = np.log10(t0_test - ts_fit[cond])
    lf_test = lf_fit[cond]

    result = linregress(x_test, lf_test)

    lf_pred = result.intercept + result.slope*x_test

    error = np.sum((lf_pred-lf_test)**2)

    if error < best_error:

        best_error = error
        t0_opt = t0_test
        alpha_opt = -result.slope
        k_opt = 10**result.intercept

print(f"Optimal t0 = {t0_opt:.3f}")
print(f"alpha = {alpha_opt:.4f}")
print(f"k = {k_opt:.4f}")

cond = ts < t0_opt

ts_model = ts[cond]

f_model = k_opt*(1/(t0_opt-ts_model))**alpha_opt

f_model_full = np.full_like(ts, f_model[-1])
f_model_full[:len(f_model)] = f_model

plt.figure(figsize=(8,4))
plt.plot(ts, fmax_smooth, label="Measured")
plt.plot(ts, f_model_full, linewidth=2, label="Power-law fit")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.grid(True)
plt.legend()
plt.show()