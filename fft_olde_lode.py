import os       
import jax
import time
import json
import optax
from flax import nnx
import jax.numpy as jnp
from pathlib import Path
from scipy.io import wavfile
import orbax.checkpoint as ocp

#jax.config.update('jax_platform_name', 'cpu')

def lode_data(split: float = 0.1, chunk_size: int = 126, spl_amt: int = 3, seed: int = 5212):
    with open("archive_1_data.csv", "r") as f:
        rows = [line.strip().split("\t") for line in f]

    images = []
    labels  = []
    distens = [] 
    num = 0 
    for i in rows:
        #if num > 40:
        #    break
        #num = num + 1

        sr, x = wavfile.read(i[2])
        x = x.astype("float32")
        x = x / 32768.0
        if spl_amt != 0:
            x = x[:len(x) // spl_amt]
        n_chunks = len(x) // chunk_size
        x = x[:n_chunks * chunk_size]
        x_spl_0 = jnp.array(x[:,0]).reshape(n_chunks, chunk_size)
        x_spl_1 = jnp.array(x[:,1]).reshape(n_chunks, chunk_size)
        
        x_fft_0 = jnp.fft.rfft(x_spl_0, axis=-1)
        x_fft_1 = jnp.fft.rfft(x_spl_1, axis=-1)
        x_fft = jnp.array([x_fft_0, x_fft_1]).flatten()
        images.append(x_fft)
        
        #print(i[0])
        angle_data = json.load(open(i[0]))
        distance_data = json.load(open(i[1]))
        labels.append(jnp.array(angle_data["LiDAR_angle"]))
        distens.append(jnp.array(distance_data["LiDAR_distance"]))
        #labels.append(int(i[0]))

    images = jnp.asarray(images, dtype=jnp.float32)
    print(jnp.shape(images))
    print(jnp.shape(distens))
    #print(jnp.shape(images[0]))
    labels = jnp.asarray(labels, dtype=jnp.float32)
    distens = jnp.asarray(distens, dtype=jnp.float32)
    unique_labels, mapped_labels = jnp.unique(labels, return_inverse=True)
    images_train, label_train, images_test, label_test = jax_train_test_split(images, distens, test_fraction=split, seed=seed)
    return images_train, label_train, images_test, label_test, mapped_labels, n_chunks

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

chunk_size = 126
#images_train, label_train, images_test, label_test, mapped_labels, n_chunks = lode_data(spl_amt=0)

#make model 
n_f = 68224
n_t = 1081
model = SimpleNN(n_features = n_f, n_targets = n_t , rngs=nnx.Rngs(0))

_, state = nnx.split(model)

checkpointer = ocp.StandardCheckpointer()
checkpoint_path = Path("/home/klyx/git/p10/P10/model_fft/1778496791/state")
state = checkpointer.restore(checkpoint_path, state)
nnx.update(model, state)

# Example input
x = jnp.ones((1, n_f), dtype=jnp.float32)

# Run data through model
print(f"{model(x)} = model({x})")
