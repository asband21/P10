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
with open("./data_set/data_set_stero_clean.csv", "r") as f:
    rows = [line.strip().split("\t") for line in f]

for i in range(0, len(rows), 2):
    if (rows[i][0] == rows[i+1][0] and rows[i][1] == rows[i+1][1] and rows[i][2] == 'h' and rows[i+1][2] == 'l')
        print(rows[i])

