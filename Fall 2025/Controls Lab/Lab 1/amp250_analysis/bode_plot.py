from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(r"C:\Users\noaht\School\Fall 2025\Controls Lab\Lab 1")
FREQ_FILE = BASE_DIR / "amp_250.xlsx"

def read_data(file_path):