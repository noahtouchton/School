from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import math

BASE_DIR = Path(r"C:\Users\noaht\School\Fall 2025\Controls Lab\Lab 1")
FREQ_FILE = BASE_DIR / "amp_250.xlsx"

def read_excel(path: Path, header_row: int) -> pd.DataFrame:
    df = pd.read_excel(path, header=header_row, engine="openpyxl")
    # Fuzzy pick columns
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        targets = [n.lower() for n in names]
        for l, orig in cols.items():
            if any(t in l for t in targets):
                return orig
        raise KeyError(f"Missing expected column like: {names}")

    col_time  = pick("Time (s)", "Time")
    col_cmd   = pick("Command Signal", "Command")
    col_angle = pick("Filtered Platen Rotation Angle", "Rotation Angle", "Platen Angle", "Filtered")
    col_vel   = pick("Platen Velocity", "Velocity")
    col_amp   = pick("Amplitude", "Amp")
    col_freq  = pick("Frequency", "Freq")

    df = df[[col_time, col_cmd, col_angle, col_vel, col_amp, col_freq]].copy()
    df.columns = ["time", "command", "angle", "velocity", "amplitude", "frequency"]

    # numeric coercion
    for c in ["time", "command", "angle", "velocity", "amplitude", "frequency"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    return df

def command_to_volts(cmd: np.ndarray, v_supply: float = 12.0) -> np.ndarray:
    return (cmd / 1023) * v_supply

def vel_to_rad_per_sec(vel: np.ndarray) -> np.ndarray:
    return (vel / 60) * 2 * np.pi

def find_frequency_blocks(freqs: np.ndarray):
    #return (start_idx, end_idx) pairs for each frequency block
    blocks = []
    if len(freqs) == 0:
        return blocks
    start_freq = freqs[0]
    start_idx = 0

    for i, f in enumerate(freqs):
        if f != start_freq:
            blocks.append((start_idx, i - 1))
            start_freq = f
            start_idx = i
    blocks.append((start_idx, len(freqs) - 1))
    return blocks

def find_amplitude_and_phase(data: pd.DataFrame, start_idx: int, end_idx: int, column: str, debug: bool=False):
    t = data["time"][start_idx:end_idx+1].to_numpy()
    y = data[column][start_idx:end_idx+1].to_numpy()

    # Use the block's set frequency
    freq = data["frequency"][start_idx]

    def sin_func(t, A, phi, offset):
        return A * np.sin(2 * np.pi * freq * t + phi) + offset

    A0 = (np.max(y) - np.min(y)) / 2
    phi0 = 0.0
    offset0 = np.mean(y)
    popt, _ = curve_fit(sin_func, t, y, p0=[A0, phi0, offset0])

    amplitude, phase, offset = popt

    # normalize amplitude sign so log10 is always happy
    if amplitude < 0:
        amplitude = -amplitude
        phase -= np.pi
    if phase > 0:
        phase *= -1
        phase -= np.pi
    if debug:
        plt.figure()
        plt.plot(t, y, 'b.', label='Data')
        t_fit = np.linspace(t.min(), t.max(), 500)
        plt.plot(t_fit, sin_func(t_fit, amplitude, phase, offset), 'r-', label='Fit')
        plt.legend()
        plt.title(f"{column} fit @ {freq:.3g} Hz")
        plt.xlabel("Time (s)")
        plt.ylabel(column.capitalize())

    return amplitude, phase


def plot_bode(k_values, phase_shifts):
    # Convert frequency from Hz to rad/s
    freqs_hz = np.array([f for f, _ in k_values])
    freqs_rad = 2 * np.pi * freqs_hz
    gains = np.array([k for _, k in k_values])
    phases = np.array([p for _, p in phase_shifts])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.semilogx(freqs_rad, gains, 'o-', label='Gain (dB)')
    ax1.set_ylabel('Gain (dB)')
    ax1.grid(True, which='both', ls='--')
    ax1.legend()

    ax2.semilogx(freqs_rad, np.degrees(phases), 'o-', label='Phase (deg)')
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.set_ylabel('Phase (deg)')
    ax2.grid(True, which='both', ls='--')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    exit()  # Stop after first Bode plot

    # Save the last Bode plot to a PNG file in the script's directory
def save_bode_plot(k_values, phase_shifts, out_path):
    freqs_hz = np.array([f for f, _ in k_values])
    freqs_rad = 2 * np.pi * freqs_hz
    gains = np.array([k for _, k in k_values])
    phases = np.array([p for _, p in phase_shifts])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.semilogx(freqs_rad, gains, 'o-', label='Gain (dB)')
    ax1.set_ylabel('Gain (dB)')
    ax1.grid(True, which='both', ls='--')
    ax1.legend()

    ax2.semilogx(freqs_rad, np.degrees(phases), 'o-', label='Phase (deg)')
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.set_ylabel('Phase (deg)')
    ax2.grid(True, which='both', ls='--')
    ax2.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# Modify main to save the plot
def main():
    df = read_excel(FREQ_FILE, header_row=3)

    df["voltage"] = command_to_volts(df["command"].to_numpy())
    df["velocity"] = vel_to_rad_per_sec(df["velocity"].to_numpy())

    blocks = find_frequency_blocks(df["frequency"].to_numpy())

    k_values = []
    phase_shifts = []
    for i, (start, end) in enumerate(blocks):
        freq = df["frequency"][start]
        amp1, phase1 = find_amplitude_and_phase(df, start, end, "voltage")
        amp2, phase2 = find_amplitude_and_phase(df, start, end, "velocity")

        if amp2/amp1 < 0:
            k = 20*math.log10(-1.0*amp2/amp1)
        else:
            k = 20*math.log10(amp2/amp1)
        phase_diff = phase2 - phase1
        k_values.append((freq, k))
        phase_shifts.append((freq, phase_diff))

    # Save Bode plot as bode.png in the script's directory
    script_dir = Path(__file__).parent
    out_path = script_dir / "bode.png"
    save_bode_plot(k_values, phase_shifts, out_path)

    KDC = (k_values[0][1] + k_values[1][1] + k_values[2][1]) / 3 #average of the first three K values
    print(f"KDC: {KDC:.2f} dB")

    

if __name__ == "__main__":
    main()
    #plt.show()