import os       
import jax
import math
import time
import json
import optax
from flax import nnx
import jax.numpy as jnp
from pathlib import Path
from scipy.io import wavfile
import orbax.checkpoint as ocp


## lode index
#with open("./data_set/data_set_2_shuffel.csv", "r") as f:
with open("./data_set_2/data_set.csv", "r") as f:
    rows = [line.strip().split("\t") for line in f]

min_fri = 20000
for i in rows:
    paft = i[3]
    distend = i[0]

    ## lode audio
    sr, x = wavfile.read("data_set_2/"+paft)
    chunk_size = math.ceil(sr / min_fri)  
    x = x.astype("float32")
    x = x / 32768.0

    n_chunks = len(x) // chunk_size
    x = x[:n_chunks * chunk_size]
    x_spl = jnp.array(x).reshape(n_chunks, chunk_size)
    x_fft = jnp.abs(jnp.fft.rfft(x_spl, axis=-1)).flatten()

    if len(x_fft) != 41668:
        print(i)
        #print("Please, yeah, the length of the audio is not the right length. The loading went wrong. Please try another one.")
        #print("data_set/"+paft)

