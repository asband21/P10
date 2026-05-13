import os       
import jax
import math
import time
import json
import optax
import alsaaudio
import subprocess
from flax import nnx
import jax.numpy as jnp
from pathlib import Path
from scipy.io import wavfile
import orbax.checkpoint as ocp

class SimpleNN(nnx.Module):
    def __init__(self, n_features: int = 19200, n_hidden: int = 255, n_targets: int = 26, *, rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(n_features, n_hidden, rngs=rngs)
        self.layer2 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer3 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer4 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.output = nnx.Linear(n_hidden, n_targets, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu6(self.layer1(x))
        x = nnx.relu6(self.layer2(x))
        x = nnx.relu6(self.layer3(x))
        x = nnx.relu6(self.layer4(x))
        x = self.output(x)
        return x


## mick config
gpio = subprocess.Popen(["sudo", "bash"], stdin=subprocess.PIPE, text=True, bufsize=1,)

gpio.stdin.write("pinctrl set 17 op dl\n")

RATE = 384000
SECONDS = 0.2
PERIODSIZE = 1024
FILE = "chunk.wav"
device_l="hw:CARD=Microphone_1,DEV=0"
device_h="hw:CARD=Microphone,DEV=0"

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


old_labels = [5, 7, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 60]
new_labels = [0, 1, 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

## lode model
n_f = 41668
n_t = 19
min_fri = 20000
model = SimpleNN(n_features = n_f, n_targets = n_t, rngs=nnx.Rngs(0), n_hidden = 2048)
nnx.display(model)  # Interactive display if penzai is installed.

_, state = nnx.split(model)

checkpointer = ocp.StandardCheckpointer()
checkpoint_path = Path("model_fft/1778674547/state").resolve()
state = checkpointer.restore(checkpoint_path, state)
nnx.update(model, state)

key = jax.random.key(int(time.time()))
char = 'y'
while(char != 'x'):
    
    chunks_l = []
    chunks_h = []
    num_reads = int(RATE * SECONDS / PERIODSIZE)

    try:
        for i in range(num_reads):
            length_l, data_l = left_mick.read()
            length_h, data_h = right_mick.read()
            if length_l > 0 and length_h > 0:
                chunks_l.append(jnp.frombuffer(data_l, dtype=jnp.int16))
                chunks_h.append(jnp.frombuffer(data_h, dtype=jnp.int16))
            if i == 1:
                gpio.stdin.write("pinctrl set 17 op dh\n")
    except:
        gpio.stdin.write("pinctrl set 17 op dl\n")
        time.sleep(0.2)
        continue
    gpio.stdin.write("pinctrl set 17 op dl\n")
    gpio.stdin.flush()

    audio_l = jnp.concatenate(chunks_l)
    audio_h = jnp.concatenate(chunks_h)
    #wavfile.write(f"data_set/chunk_({device_l})_{post_fix}_{iii}_l.wav", RATE, audio_l)
    #wavfile.write(f"data_set/chunk_({device_l})_{post_fix}_{iii}_h.wav", RATE, audio_h)
    x = audio_l

    ## lode audio
    ##sr, x = wavfile.read("data_set_2/"+paft)

    chunk_size = math.ceil(RATE / min_fri)  
    x = x.astype("float32")
    x = x / 32768.0

    n_chunks = len(x) // chunk_size
    x = x[:n_chunks * chunk_size]
    x_spl = jnp.array(x).reshape(n_chunks, chunk_size)
    x_fft = jnp.abs(jnp.fft.rfft(x_spl, axis=-1)).flatten()

    if len(x_fft) != 41668:
        print("Please, yeah, the length of the audio is not the right length. The loading went wrong. Please try another one.")
        print(audio_l)
        print(jnp.shape(audio_l))
        continue
        os.exit(1)

    ## run pridick
    prediksen = model(x_fft)
    #print(f"{model(x_fft)} = model({x_fft})")
    index = jnp.argmax(prediksen)
    print(f"pridiksen {old_labels[index]}")
    
    time.sleep(0.1)
    #char = input("tryk 'x' for a stoppe:")
