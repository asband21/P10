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
SECONDS = 0.1
PERIODSIZE = 1024
FILE = "chunk.wav"

device="two_mics"

micks = alsaaudio.PCM(
    type=alsaaudio.PCM_CAPTURE,
    device=device,
    channels=2,
    rate=RATE,
    format=alsaaudio.PCM_FORMAT_S16_LE,
    periodsize=PERIODSIZE,
)


for iii in range(100):
    chunks = []
    num_reads = int(RATE * SECONDS / PERIODSIZE)
    
    for _ in range(4):
        micks.read()
    
    #try:
    lgpio.gpio_write(h, GPIO_LINE, 1)
    for i in range(num_reads):
        length, data = micks.read()
        chunks.append(np.frombuffer(data, dtype=np.int16))
        #chunks.append(data)
            #if length > 0:
            #    chunks.append(np.frombuffer(data, dtype=np.int16))
    #except:
    #    lgpio.gpio_write(h, GPIO_LINE, 0)
    #    print(iii)
    #    time.sleep(0.2)
    #    continue
    # Combine raw ALSA byte buffers after recording
    #raw_audio = b"".join(chunks)
    # Convert after recording, not inside the read loop
    #audio = np.frombuffer(raw_audio, dtype=np.int16)
    #audio = audio.reshape(-1, 2)
    # gpio lown
    #chunks = np.array(chunks, dtype=np.int16)

    lgpio.gpio_write(h, GPIO_LINE, 0)
    audio = np.concatenate(chunks)
    audio = audio.reshape(-1, 2)
    wavfile.write(f"data_set_gpio/chunk_({device})_{post_fix}_{iii}_stero.wav", RATE, audio)
    time.sleep(0.2)

micks.close()
lgpio.gpiochip_close(h)


#subprocess.run(["bash", "spektrum.sh"], check=True)
