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

            t_start = 22.0
            t_end = 26.0

            sweep_mask = (times >= t_start) & (times <= t_end)
            t_sweep = times[sweep_mask]
            omega_sweep = peak_frequencies[sweep_mask]

            t_f = t_end + 0.05

            log_delta_t = np.log10(t_f - t_sweep)
            log_omega = np.log10(omega_sweep)

            # --- THEORETICAL SLOPES (n_omega = n_theta / 2) ---
            slope_roll = -1 / 3  # Rolling Friction (n = 0.33)
            slope_moffatt = -1 / 6  # Moffatt Air (n = 0.17)
            slope_bl = -2 / 9  # Boundary Layer (n = 0.22)

            # Anchor the theoretical lines to the center of the experimental data
            mean_x = np.mean(log_delta_t)
            mean_y = np.mean(log_omega)

            int_roll = mean_y - slope_roll * mean_x
            int_moffatt = mean_y - slope_moffatt * mean_x
            int_bl = mean_y - slope_bl * mean_x

            fit_roll = slope_roll * log_delta_t + int_roll
            fit_moffatt = slope_moffatt * log_delta_t + int_moffatt
            fit_bl = slope_bl * log_delta_t + int_bl

            # --- PLOTTING ---
            plt.figure(figsize=(8, 5))

            # Experimental Data
            plt.plot(
                log_delta_t,
                log_omega,
                "k.",
                markersize=6,
                label="Experimental Data (f)",
            )

            # Theoretical Lines
            plt.plot(
                log_delta_t,
                fit_roll,
                "r--",
                alpha=0.8,
                label="Rolling Friction (n = 1/3)",
            )
            plt.plot(
                log_delta_t, fit_bl, "g--", alpha=0.8, label="BL (n = 2/9)"
            )
            plt.plot(
                log_delta_t,
                fit_moffatt,
                "b--",
                alpha=0.8,
                label="Moffatt Air (n = 1/6)",
            )

            plt.title(
                f"Precession Rate vs Theoretical Regimes ({os.path.basename(root)})"
            )
            plt.xlabel(r"$\log_{10}(t_f - t)$")
            plt.ylabel(r"$\log_{10}(f)$")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()

            plt.tight_layout()
            save_path = os.path.join(root, "bilog_theory_comparison.png")
            plt.savefig(save_path, dpi=300)
            plt.close()

            print(f"Theory plot successfully saved for {os.path.basename(root)}")
