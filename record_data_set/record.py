import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    if status:
        print(status)
    volume = np.sqrt(np.mean(indata**2))
    print(volume)

with sd.InputStream(channels=1, samplerate=96000, callback=callback):
    input("Tryk Enter for at stoppe...\n")
