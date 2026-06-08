# ./notebooks/test.py
import os

from euler_disk import short_time_ft

input_dir = "../data/wav/"
data_files = sorted([file for file in os.listdir(input_dir) if file.endswith(".wav")])

for measurements in data_files:
    file_name = measurements.replace(".wav", "")
    output_dir = f"../data/output/{file_name}/"

    os.makedirs(output_dir, exist_ok=True)
    short_time_ft(f"{input_dir}{file_name}.wav", output_dir)
    print(f"Data sucessfully processed for {file_name}.")
