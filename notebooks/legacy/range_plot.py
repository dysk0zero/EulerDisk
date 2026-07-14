import os

import matplotlib.pyplot as plt
import numpy as np

base_dir = "../data/output/"

for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".npz"):
            data_path = os.path.join(root, file)
            data = np.load(data_path)

            frequencies = data["frequencies"]
            times = data["times"]
            magnitude = data["magnitude"]

            valid_indices = np.where((frequencies >= 20) & (frequencies <= 300))[0]
            filtered_freqs = frequencies[valid_indices]
            filtered_mag = magnitude[valid_indices, :]

            peak_indices = np.argmax(filtered_mag, axis=0)
            peak_frequencies = filtered_freqs[peak_indices]

            plt.figure(figsize=(8, 5))
            plt.plot(times, peak_frequencies, "k.", markersize=2)

            plt.title("Precession Frequency vs. Time")
            plt.xlabel("Time (s)")
            plt.ylabel(r"$\Omega$ (Hz)")
            plt.grid(True, linestyle="--", alpha=0.6)

            plt.tight_layout()
            save_path = os.path.join(root, "precession_curve.png")
            plt.savefig(save_path, dpi=300)
            plt.close()

            print(f"Plot successfully saved to {save_path}")
