import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

# Load audio
fs, x = wavfile.read("data_set/chunk_(hw:CARD=Microphone_1,DEV=0)_23cm_27_l.wav")

# If stereo, use one channel
if x.ndim > 1:
    x = x[:, 0]

# Convert integer audio to float
x = x.astype(np.float32)
x = x / np.max(np.abs(x))

# STFT settings
n_fft = 2048        # FFT size
hop_length = 512    # step between windows
window_length = 2048

# Compute STFT
freqs, times, Zxx = stft(
    x,
    fs=fs,
    window="hann",
    nperseg=window_length,
    noverlap=window_length - hop_length,
    nfft=n_fft,
    boundary=None
)

# Magnitude in dB
S_db = 20 * np.log10(np.abs(Zxx) + 1e-12)

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(
    S_db,
    aspect="auto",
    origin="lower",
    extent=[times[0], times[-1], freqs[0], freqs[-1]]
)

plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrogram")
plt.colorbar(label="Magnitude [dB]")
plt.show()
