import alsaaudio
import numpy as np
from scipy.io import wavfile

RATE = 200000
FILE = "chunk.wav"

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
inp.setchannels(1)
inp.setrate(RATE)
inp.setformat(alsaaudio.PCM_FORMAT_S32_LE)
inp.setperiodsize(1024)

l, d = inp.read()
inp.close()

audio = np.frombuffer(d, dtype=np.int32)
wavfile.write(FILE, RATE, audio)
