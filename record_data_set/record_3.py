import alsaaudio
import numpy as np
from scipy.io import wavfile

RATE = 200000
SECONDS = 5
PERIODSIZE = 1024
FILE = "chunk.wav"

inp = alsaaudio.PCM(
    type=alsaaudio.PCM_CAPTURE,
    channels=1,
    rate=RATE,
    format=alsaaudio.PCM_FORMAT_S32_LE,
    periodsize=PERIODSIZE,
)

chunks = []

num_reads = int(RATE * SECONDS / PERIODSIZE)

for _ in range(num_reads):
    length, data = inp.read()
    if length > 0:
        chunks.append(np.frombuffer(data, dtype=np.int32))

inp.close()

audio = np.concatenate(chunks)
wavfile.write(FILE, RATE, audio)
