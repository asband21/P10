import alsaaudio
import numpy as np
import threading
import sys

# Settings
CHANNELS = 1
RATE = 96000
FORMAT = alsaaudio.PCM_FORMAT_S16_LE  # 16-bit little-endian
PERIOD_SIZE = 1024  # frames per read

# Open PCM device for capture
inp = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL)
inp.setchannels(CHANNELS)
inp.setrate(RATE)
inp.setformat(FORMAT)
inp.setperiodsize(PERIOD_SIZE)

running = True

def recorder():
    global running
    while running:
        l, data = inp.read()
        if l:
            # Convert byte data to numpy array of int16
            samples = np.frombuffer(data, dtype=np.int16)
            # If stereo and CHANNELS>1, reshape and take first channel
            if CHANNELS > 1:
                samples = samples.reshape(-1, CHANNELS)[:, 0]
            # Normalize to -1..1 float
            samples_f = samples.astype(np.float32) / 32768.0
            volume = np.sqrt(np.mean(samples_f**2))
            print(volume)
        else:
            # No data; small sleep to avoid busy loop
            import time
            time.sleep(0.01)

t = threading.Thread(target=recorder, daemon=True)
t.start()

try:
    input("Tryk Enter for at stoppe...\n")
finally:
    running = False
    t.join(timeout=1)
    inp.close()
