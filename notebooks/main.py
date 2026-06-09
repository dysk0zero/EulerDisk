# ./notebooks/test.py
import os

from euler_disk import short_time_ft

input_dir = "../data/wav/"
data_files = sorted([file for file in os.listdir(input_dir) if file.endswith(".wav")])

# Calculates the max vale of 2^n for the window size in the STFT
# A good range for the STFT would be between n=10 and n=16
max_power_window = 15
min_power_window = 11

for i in range(min_power_window, max_power_window + 1):
    window_size = 2**i
    hop_size = 2 ** (i - 3)
    for measurements in data_files:
        file_name = measurements.replace(".wav", "")
        output_dir = f"../data/output/window_{window_size}/{file_name}/"

        os.makedirs(output_dir, exist_ok=True)
        short_time_ft(
            f"{input_dir}{file_name}.wav",
            output_dir,
            window_size=window_size,
            hop_size=hop_size,
        )
        print(f"Data sucessfully processed at window={window_size} for {file_name}.")
