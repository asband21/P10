import json
import sys
import jax
import time
from flax import nnx
import jax.numpy as jnp
from pathlib import Path
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

from scipy.io import wavfile

def to_cartesian_scalar(a, d):
    return jnp.cos(a)*d

def to_cartesian(polar):
    x = jnp.cos(polar[:,0])*polar[:,1]
    y = jnp.sin(polar[:,0])*polar[:,1]
    return jnp.array([x,y]).T

class SimpleNN(nnx.Module):
    def __init__(self, n_features: int = 19200, n_hidden: int = 255, n_targets: int = 26, *, rngs: nnx.Rngs):
        self.n_features = n_features
        self.layer1 = nnx.Linear(int(n_features), n_hidden, rngs=rngs)
        self.layer2 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer3 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer4 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer5 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer6 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.output = nnx.Linear(n_hidden, n_targets, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu6(self.layer1(x))
        x = nnx.relu6(self.layer2(x))
        x = nnx.relu6(self.layer3(x))
        x = nnx.relu6(self.layer4(x))
        x = nnx.relu6(self.layer5(x))
        x = nnx.relu6(self.layer6(x))
        x = self.output(x)
        return x

angle_path = ""
distance_path = ""
sound_path = ""

# In the case where we want to load the models directly, we have to uncomment this.
if False:
    if len(sys.argv) != 4:
        print("Command must be in this form:\n script.py Wide_Long_angle_data.json Wide_Long_distance_data.json Wide_Long_sound.wav")
        sys.exit(1)
    angle_path = sys.argv[1]
    distance_path = sys.argv[2]
    sound_path = sys.argv[3]

else:
    with open("archive_1_data.csv", "r") as f:
        rows = [line.strip().split("\t") for line in f]
    key = jax.random.key(int(time.time()))
    idx = jax.random.randint(key, shape=(), minval=0, maxval=len(rows)-1)
    #print("idx:"+ str(idx) + "\t" + rows[idx][0] + "\t" +rows[idx][1] + "\t" + rows[idx][2])
    angle_path = rows[idx][0]
    distance_path = rows[idx][1]
    sound_path = rows[idx][2]

## Load data
angle = json.load(open(angle_path))
distance = json.load(open(distance_path))
sr, x = wavfile.read(sound_path)

#lode audio
x = x.astype("float32")
x = x / 32768.0
spl_amt = 3
chunk_size = 126
x = x[:len(x) // spl_amt]
n_chunks = len(x) // chunk_size
x = x[:n_chunks * chunk_size]
x_spl_0 = jnp.array(x[:,0]).reshape(n_chunks, chunk_size)
x_spl_1 = jnp.array(x[:,1]).reshape(n_chunks, chunk_size)
x_fft_0 = jnp.fft.rfft(x_spl_0, axis=-1)
x_fft_1 = jnp.fft.rfft(x_spl_1, axis=-1)
x_fft = jnp.array([x_fft_0, x_fft_1], dtype=jnp.float32).flatten()

polar_data = jnp.array([angle["LiDAR_angle"], distance["LiDAR_distance"]]).T

## lode model 
n_f = 68224
n_t = 1081
model = SimpleNN(n_features = n_f, n_targets = n_t , rngs=nnx.Rngs(0))

_, state = nnx.split(model)

checkpointer = ocp.StandardCheckpointer()
checkpoint_path = Path("model_fft/1778496791/state")
state = checkpointer.restore(checkpoint_path, state)
nnx.update(model, state)

## model pridiksin
model_data = model(x_fft)

## Transform data
cartesian_data = to_cartesian(polar_data)
cartesian_model_data = to_cartesian(model_data)

## Gradients
#grad_to_cartesian = jax.jacobian(to_cartesian)
#grad_to_cartesian_scalar = jax.grad(to_cartesian_scalar)

#g = grad_to_cartesian_scalar(1.0, 1.0)
#grad_data = grad_to_cartesian(polar_data)

#sys.exit(1)

## Plot data
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
