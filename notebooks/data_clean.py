import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d


# -------------------------------------------------------
# Parameters
# -------------------------------------------------------

input_file = "../data/raw/huge_enourmous_disc_dingly_dongus_recording_five_times.wav"
output_dir = "../data/split"

os.makedirs(output_dir, exist_ok=True)

threshold_ratio = 0.08      # Fraction of maximum envelope
smooth_ms = 100             # Envelope smoothing window
min_duration = 8.0          # Minimum trial duration (s)
padding = 0.5               # Seconds before/after each trial
merge_gap = 1.0             # Merge regions separated by less than this

start_offset = 2.0 

# -------------------------------------------------------
# Load audio
# -------------------------------------------------------

fs, x = wavfile.read(input_file)

if x.ndim > 1:
    x = x[:, 0]

x_out = x.copy()

x = x.astype(np.float32)


# -------------------------------------------------------
# Envelope
# -------------------------------------------------------

env = np.abs(x)

window = int(fs * smooth_ms / 1000)

env = uniform_filter1d(env, size=window)

threshold = threshold_ratio * env.max()/2

active = env > threshold


# -------------------------------------------------------
# Find active regions
# -------------------------------------------------------

changes = np.diff(active.astype(int))

starts = np.where(changes == 1)[0]
stops = np.where(changes == -1)[0]

if active[0]:
    starts = np.insert(starts, 0, 0)

if active[-1]:
    stops = np.append(stops, len(active) - 1)


# -------------------------------------------------------
# Merge nearby regions
# -------------------------------------------------------

merged = []

current_start = starts[0]
current_stop = stops[0]

for s, e in zip(starts[1:], stops[1:]):

    gap = (s - current_stop) / fs

    if gap < merge_gap:
        current_stop = e
    else:
        merged.append((current_start, current_stop))
        current_start = s
        current_stop = e

merged.append((current_start, current_stop))


# -------------------------------------------------------
# Keep only long enough recordings
# -------------------------------------------------------

regions = []

for start, stop in merged:

    duration = (stop - start) / fs

    if duration >= min_duration:
        regions.append((start, stop))


print(f"Detected {len(regions)} recordings.")


# -------------------------------------------------------
# Save
# -------------------------------------------------------

pad = int(fs * padding)

for i, (start, stop) in enumerate(regions, start=1):

    a = max(0, start + int(start_offset * fs))
    b = min(len(x), stop + pad)

    wavfile.write(
        os.path.join(output_dir, f"big_{i:02d}.wav"),
        fs,
        x_out[a:b],
    )


# -------------------------------------------------------
# Visualize
# -------------------------------------------------------

t = np.arange(len(x)) / fs

plt.figure(figsize=(14,5))

plt.plot(t, env, label="Envelope")

plt.axhline(threshold, color="red", ls="--", label="Threshold")

for start, stop in regions:

    plt.axvspan(start/fs, stop/fs, color="green", alpha=0.3)

plt.xlabel("Time (s)")
plt.ylabel("Envelope")
plt.title("Detected Trials")
plt.legend()

plt.show()