import time
import numpy as np
import alsaaudio
import subprocess
from scipy.io import wavfile



# gpio high
print("set high")
subprocess.run(["sudo", "pinctrl", "set", "17", "op", "dl"], check=True)
subprocess.run(["sudo", "pinctrl", "set", "17", "op", "dh"], check=True)



RATE = 200000
SECONDS = 0.1
PERIODSIZE = 1024
FILE = "chunk.wav"

inp = alsaaudio.PCM(
    type=alsaaudio.PCM_CAPTURE,
    channels=1,
    rate=RATE,
    format=alsaaudio.PCM_FORMAT_S32_LE,
    periodsize=PERIODSIZE,
)

for iii in range(10):
    chunks = []
    num_reads = int(RATE * SECONDS / PERIODSIZE)

    subprocess.run(["sudo", "pinctrl", "set", "17", "op", "dh"], check=True)
    for _ in range(num_reads):
        length, data = inp.read()
        if length > 0:
            chunks.append(np.frombuffer(data, dtype=np.int32))

    subprocess.run(["sudo", "pinctrl", "set", "17", "op", "dl"], check=True)
    # gpio lown

    audio = np.concatenate(chunks)
    wavfile.write(f"data_set/chunk_{iii}.wav", RATE, audio)

subprocess.run(["bash", "spektrum.sh"], check=True)
inp.close()
