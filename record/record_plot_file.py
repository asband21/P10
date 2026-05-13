import alsaaudio
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

RATE = 200000
sr, x = wavfile.read("chunk.wav")


audio = np.frombuffer(x, dtype=np.int64)

fft = np.fft.rfft(audio)
freqs = np.fft.rfftfreq(len(audio), 1 / RATE)

plt.plot(freqs, np.abs(fft))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.show()
