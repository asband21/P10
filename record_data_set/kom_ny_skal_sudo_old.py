import sys
import time
import lgpio
import numpy as np
#import jax.numpy as jnp
import alsaaudio
import subprocess
from scipy.io import wavfile


GPIO_CHIP = 0      # often gpiochip4 on Raspberry Pi 5, but check with gpioinfo
GPIO_LINE = 17

h = lgpio.gpiochip_open(GPIO_CHIP)
lgpio.gpio_claim_output(h, GPIO_LINE, 0)
lgpio.gpio_write(h, GPIO_LINE, 0)


post_fix = ""
print(sys.argv)
if len(sys.argv) == 2:
    post_fix = sys.argv[1] 
    print(f" post fix {post_fix}")

RATE = 384000
SECONDS = 0.2
PERIODSIZE = 1024
PERIODSIZE = 256
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


for iii in range(10):
    chunks_l = []
    chunks_h = []
    num_reads = int(RATE * SECONDS / PERIODSIZE)
    
    for _ in range(10):
        left_mick.read()
        right_mick.read()
    
    try:
        
        for i in range(num_reads):
            length_l, data_l = left_mick.read()
            length_h, data_h = right_mick.read()
            if length_l > 0:
                chunks_l.append(np.frombuffer(data_l, dtype=np.int16))
            if length_h > 0:
                chunks_h.append(np.frombuffer(data_h, dtype=np.int16))
            if i == 1:
                lgpio.gpio_write(h, GPIO_LINE, 1)
    except:
        lgpio.gpio_write(h, GPIO_LINE, 0)
        print(iii)
        time.sleep(0.2)
        continue

    lgpio.gpio_write(h, GPIO_LINE, 0)
    # gpio lown

    audio_l = np.concatenate(chunks_l)
    audio_h = np.concatenate(chunks_h)
    wavfile.write(f"data_set_gpio/chunk_({device_l})_{post_fix}_{iii}_l.wav", RATE, audio_l)
    wavfile.write(f"data_set_gpio/chunk_({device_l})_{post_fix}_{iii}_h.wav", RATE, audio_h)
    time.sleep(0.2)

left_mick.close()
right_mick.close()
lgpio.gpiochip_close(h)


#subprocess.run(["bash", "spektrum.sh"], check=True)
