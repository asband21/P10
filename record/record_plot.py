import alsaaudio
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt



RATE = 200000

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
inp.setchannels(1)
inp.setrate(RATE)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
inp.setperiodsize(1024)

l, d = inp.read()
inp.close()

audio = np.frombuffer(d, dtype=np.int16)

fft = np.fft.rfft(audio)
freqs = np.fft.rfftfreq(len(audio), 1 / RATE)

plt.plot(freqs, np.abs(fft))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.show()
