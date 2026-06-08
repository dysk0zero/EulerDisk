# ./src/data_processing/utils.py
import numpy as np
from scipy.io import wavfile


def extract_data(input_path: str, output_path: str) -> None:
    rate, data = wavfile.read(input_path)
    print("Data Rate: ", rate, "\n")
    return None
