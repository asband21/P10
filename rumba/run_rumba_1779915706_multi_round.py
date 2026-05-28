import os       
import jax
import math
import time
import json
import optax
#import numpy 
from flax import nnx
import jax.numpy as jnp
from pathlib import Path
from scipy.io import wavfile
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

def to_cartesian(polar):
    x = jnp.cos(polar[:,0])*polar[:,1]
    y = jnp.sin(polar[:,0])*polar[:,1]
    return jnp.array([x,y]).T

class cnn(nnx.Module):
    def __init__(self, n_in_len: int = 300, n_in_hite: int = 64, n_in_deeb: int = 2, n_hidden: int = 1024, n_targets: int = 360, kn_size: int = 5,  *, rngs: nnx.Rngs):
        self.layer1 = nnx.Conv(in_features=n_in_deeb, out_features=32, kernel_size=(5,5),  padding='VALID', rngs=rngs)
        self.layer2 = nnx.Conv(in_features=32, out_features=32, kernel_size=(5,5),  padding='VALID', rngs=rngs)
        self.layer3 = nnx.Conv(in_features=32, out_features=32, kernel_size=(5,5),  padding='VALID', rngs=rngs)
        self.layer4 = nnx.Conv(in_features=32, out_features=32, kernel_size=(5,5),  padding='VALID', rngs=rngs)
        self.layer5 = nnx.Conv(in_features=32, out_features=1, kernel_size=(5,5),  padding='VALID', rngs=rngs)
        self.layer6 = nnx.Linear(13629, n_hidden, rngs=rngs)
        self.layer7 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer8 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.output = nnx.Linear(n_hidden, n_targets, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu6(self.layer1(x))
        x = nnx.relu6(self.layer2(x))
        x = nnx.relu6(self.layer3(x))
        x = nnx.relu6(self.layer4(x))
        x = nnx.relu6(self.layer5(x))
        x = x.reshape((x.shape[0], -1))
        x = nnx.relu6(self.layer6(x))
        x = nnx.relu6(self.layer7(x))
        x = nnx.relu6(self.layer8(x))
        x = self.output(x)
        return x

angels = []
for i in range(360):
    angels.append(3.14*float(i)/180)
angels = jnp.array(angels)

n_f = 41668
n_t = 360
min_fri = 2000

## inishilinge model 
model = cnn(n_in_len = 197, n_in_hite = 97, n_targets=n_t, rngs=nnx.Rngs(0))
nnx.display(model)  # Interactive display if penzai is installed.
## lode model
_, state = nnx.split(model)
checkpointer = ocp.StandardCheckpointer()
checkpoint_path = Path("./model_cnn_rumba/1779915706/state_last").resolve()
state = checkpointer.restore(checkpoint_path, state)
nnx.update(model, state)

## lode index
rows = []
for i in range(2, 11):
    with open(f"data_set_rumba/data_set_rumba_{i}/master_dataset_index.csv", "r") as f:
        #print(f)
        rows_m = [line.strip().split("\t") for line in f]
    for lolo in rows_m:
        lolo[4] = f"data_set_rumba/data_set_rumba_{i}/" + lolo[4]
        lolo[5] = f"data_set_rumba/data_set_rumba_{i}/" + lolo[5]
    rows.extend(rows_m)

key = jax.random.key(int(time.time()))
char = 'y'
while(char != 'x'):
    key, k = jax.random.split(key)
    r_i = jax.random.randint(k, shape=(), minval=0, maxval=len(rows))

    paft_imges = rows[r_i][5]
    paft_lidar = rows[r_i][4]

    ## lode audio
    sr, audio_f = wavfile.read(paft_imges)                                                                                                                                                                         
    if len(audio_f) != 37888:
        print(len(audio_f))
        os.exit(1)

    chunk_size = math.ceil(sr / min_fri)  
    audio_f = audio_f.astype("float32")
    audio_f = audio_f / 32768.0

    r = audio_f[:, 0]
    l = audio_f[:, 1]
    n_chunks_r = len(r) // chunk_size
    n_chunks_l = len(l) // chunk_size
    r = r[:n_chunks_r * chunk_size]
    l = l[:n_chunks_l * chunk_size]
    r_spl = jnp.array(r).reshape(n_chunks_r, chunk_size)
    l_spl = jnp.array(l).reshape(n_chunks_l, chunk_size)
    r_fft = jnp.abs(jnp.fft.rfft(r_spl, axis=-1)) #.flatten()
    l_fft = jnp.abs(jnp.fft.rfft(l_spl, axis=-1)) #.flatten()

    image = jnp.array([r_fft, l_fft])
    image = image[None, ...]
    image = jnp.transpose(image, (0, 2, 3, 1))   # shape: (1, chunks, fft_bins, 2)

    ## lode lidar
    lidar_data = []
    with open(paft_lidar, "r") as f:
        lidar_data = [ float(line) for line in f]
    lidar = jnp.array(lidar_data)

    ## run pridick
    prediksen = model(image)
    #print(f"{prediksen} = model({image})")
    index = jnp.argmax(prediksen)

    ## zip data
    polar_data = jnp.array([angels, lidar]).T
    prediksen = jnp.squeeze(prediksen) # shape: (360,)
    model_data = jnp.array([angels, prediksen]).T

    ## Transform data
    cartesian_data = to_cartesian(polar_data)
    cartesian_model_data = to_cartesian(model_data)

    x_vals = jnp.array(cartesian_data[:, 0])
    y_vals = jnp.array(cartesian_data[:, 1])
    x_vals_model = jnp.array(cartesian_model_data[:, 0])
    y_vals_model = jnp.array(cartesian_model_data[:, 1])

    plt.figure()
    plt.scatter(x_vals, y_vals)
    plt.scatter(x_vals_model, y_vals_model)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("LiDAR XY Plot")
    plt.axis("equal")
    plt.show()
    #char = input("tryk 'x' for a stoppe:")
