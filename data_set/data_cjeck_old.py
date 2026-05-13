import jax
import optax
from flax import nnx
import jax.numpy as jnp
from scipy.io import wavfile

with open("data_set_3.csv", "r") as f:
    rows = [line.strip().split("\t") for line in f]
bukkita = [[0,0]]
num = 0
for i in rows:
    #print(i)
    #num = num + 1
    #if 30 < num:
    #    break
    sr, x = wavfile.read(i[3])
    if len(x) != 113664:
        print(i)
#print("idx:"+ str(idx) + "\t" + rows[idx][0] + "\t" +rows[idx][1] + "\t" + rows[idx][2])
