import json
import sys
import jax
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp

from scipy.io import wavfile

groop = ""
groop_num = ""
mic_ret = ""
sound_path = ""
frekvensy = 384000
period_size = 1024

# In the case where we want to load the models directly, we have to uncomment this.
if False:
    if len(sys.argv) != 5:
        print("Command must be in this form:\n script.py groop groop_num mic_ret sound_path")
        print("eksampel: python distens_sens.py 11cm    2   h	\'data_set/chunk_(hw:CARD=Microphone_1,DEV=0)_11cm_2_h.wav\'")
        sys.exit(1)
    groop = sys.argv[1]
    groop_num = sys.argv[2] 
    mic_ret = sys.argv[3]
    sound_path = sys.argv[3]

else:
    with open("data_set/data_set.csv", "r") as f:
        rows = [line.strip().split("\t") for line in f]
    key = jax.random.key(int(time.time()))
    idx = jax.random.randint(key, shape=(), minval=0, maxval=len(rows)-2) + 1
    #print("idx:"+ str(idx) + "\t" + rows[idx][0] + "\t" +rows[idx][1] + "\t" + rows[idx][2])
    groop = rows[idx][0]
    groop_num = rows[idx][1]
    mic_ret = rows[idx][2]
    sound_path = "data_set/" + rows[idx][3]

## Load data
print(sound_path)
sr, x = wavfile.read(sound_path)
x_jnp = jnp.array(x)

print(sr)
print(x)
print(len(x))
print(x[0])


jnp.fft.fftfreq(1024 ,384000)

#x_spl = jnp.split(x_jnp, int(384000/1000))
#x_spl = x_jnp.reshape(-1, 1000)
#print(jnp.shape(x_spl))
#x_spl = x_spl[:-1]
#x_spl = jnp.stack(x_spl)    # convert list of chunks into one JAX array
#print(x_spl.shape)
#x_ffts = jnp.fft.fft(x_spl, axis=1)
#print(x_ffts[0])

sys.exit(1)
## Plot data
x_vals = jnp.array(cartesian_data[:, 0])
y_vals = jnp.array(cartesian_data[:, 1])

plt.figure()
plt.scatter(x_vals, y_vals)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("LiDAR XY Plot")
plt.axis("equal")
plt.show()
