# ./src/euler_data_processing/data_processing.py
import numpy as np
from scipy import signal
from scipy.io import wavfile


def short_time_ft(input_path: str, output_path: str) -> None:
    rate, data = wavfile.read(input_path)
