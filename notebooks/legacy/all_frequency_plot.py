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

            magnitude_db = 20 * np.log10(magnitude + 1e-10)

            plt.figure(figsize=(10, 6))
            pcm = plt.pcolormesh(
                times, frequencies, magnitude_db, cmap="inferno", shading="auto"
            )

            plt.title("Euler Disk Frequency vs Time (Spectrogram)")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.ylim(0, 20000)

            cbar = plt.colorbar(pcm)
            cbar.set_label("Intensity (dB)")

            plt.tight_layout()

            save_path = os.path.join(root, "spectrogram.png")
            plt.savefig(save_path, dpi=300)
            plt.close()

            print(f"Plot successfully saved to {save_path}")
