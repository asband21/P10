import sys
import time
import numpy as np
import alsaaudio
import subprocess
from scipy.io import wavfile

gpio = subprocess.Popen(
    ["sudo", "bash"],
    stdin=subprocess.PIPE,
    text=True,
    bufsize=1,
)

##subprocess.run(["sudo", "pinctrl", "set", "17", "op", "dl"], check=Trueg
gpio.stdin.write("pinctrl set 17 op dl\n")

post_fix = ""
print(sys.argv)
if len(sys.argv) == 2:
    post_fix = sys.argv[1] 
    print(f" post fix {post_fix}")

RATE = 384000
SECONDS = 0.3
PERIODSIZE = 1024
FILE = "chunk.wav"

device_l=""
device_l="hw:CARD=Microphone_1,DEV=0"
device_h="hw:CARD=Microphone,DEV=0"
#device_l="plughw:CARD=Microphone,DEV=0"
#device_l="plughw:CARD=Microphone_1,DEV=0"
#device_l="sysdefault:CARD=Microphone_1"
#device_l="front:CARD=Microphone_1,DEV=0"
#device_l="dsnoop:CARD=Microphone_1,DEV=0"
#device_l="hw:CARD=Microphone,DEV=0"
#device_l="plughw:CARD=Microphone,DEV=0"
#device_l="sysdefault:CARD=Microphone"
#device_l="front:CARD=Microphone,DEV=0"
#device_l="dsnoop:CARD=Microphone,DEV=0"


left_mick = alsaaudio.PCM(
    type=alsaaudio.PCM_CAPTURE,
    device=device_l,
    channels=1,
    rate=RATE,
    format=alsaaudio.PCM_FORMAT_S16_LE,
    periodsize=PERIODSIZE,
)

right_mick = alsaaudio.PCM(
    type=alsaaudio.PCM_CAPTURE,
    device=device_h,
    channels=1,
    rate=RATE,
    format=alsaaudio.PCM_FORMAT_S16_LE,
    periodsize=PERIODSIZE,
)


for iii in range(50):
    chunks_l = []
    chunks_h = []
    num_reads = int(RATE * SECONDS / PERIODSIZE)

    for i in range(num_reads):
        length_l, data_l = left_mick.read()
        length_h, data_h = right_mick.read()
        if length_l > 0 and length_h > 0:
            chunks_l.append(np.frombuffer(data_l, dtype=np.int16))
            chunks_h.append(np.frombuffer(data_h, dtype=np.int16))
        if i == 2:
            #subprocess.run(["sudo", "pinctrl", "set", "17", "op", "dh"], check=True)
            gpio.stdin.write("pinctrl set 17 op dh\n")
        #if i == 15:
    #subprocess.run(["sudo", "pinctrl", "set", "17", "op", "dl"], check=True)
    gpio.stdin.write("pinctrl set 17 op dl\n")
    gpio.stdin.flush()
    # gpio lown

    audio_l = np.concatenate(chunks_l)
    audio_h = np.concatenate(chunks_h)
    wavfile.write(f"data_set/chunk_({device_l})_{post_fix}_{iii}_l.wav", RATE, audio_l)
    wavfile.write(f"data_set/chunk_({device_l})_{post_fix}_{iii}_h.wav", RATE, audio_h)
    time.sleep(0.2)

left_mick.close()
right_mick.close()
#subprocess.run(["bash", "spektrum.sh"], check=True)
