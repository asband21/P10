import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

fs = 384000       # sample rate in Hz, change if needed
chunk_size = 1000

# Magnitude of FFT
x_mag = jnp.abs(x_ffts)

# Keep only positive frequencies
x_mag = x_mag[:, :chunk_size // 2]

# Convert to dB
x_db = 20 * jnp.log10(x_mag + 1e-12)

# Frequency axis
freqs = jnp.fft.fftfreq(chunk_size, d=1/fs)[:chunk_size // 2]

# Time axis
times = jnp.arange(x_ffts.shape[0]) * chunk_size / fs

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(
    x_db.T,
    aspect="auto",
    origin="lower",
    extent=[
        float(times[0]),
        float(times[-1]),
        float(freqs[0]),
        float(freqs[-1])
    ]
)

plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("FFT magnitude over time")
plt.colorbar(label="Magnitude [dB]")
plt.show()
