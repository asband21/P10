import os       
import jax
import time
import optax
from flax import nnx
import jax.numpy as jnp
from pathlib import Path
from scipy.io import wavfile
import orbax.checkpoint as ocp

def lode_data(split: float = 0.1, chunk_size: int = 126, spl_amt: int = 3, seed: int = 5212):
    with open("../data_set/data_set_3.csv", "r") as f:
        rows = [line.strip().split("\t") for line in f]
    images = []
    labels  = []
    for i in rows:
        sr, x = wavfile.read("../data_set/" + i[3])
        x = x.astype("float32")
        x = x / 32768.0
        x = x[:len(x) // spl_amt]
        n_chunks = len(x) // chunk_size
        x = x[:n_chunks * chunk_size]
        x_spl = jnp.array(x).reshape(n_chunks, chunk_size)

        x_fft = jnp.fft.rfft(x_spl, axis=-1)
        x_fft = x_fft.flatten()
        images.append(x_fft)
        labels.append(int(i[0]))

    images = jnp.asarray(images, dtype=jnp.float32)
    print(jnp.shape(images))
    print(jnp.shape(images[0]))
    labels = jnp.asarray(labels, dtype=jnp.int32)
    unique_labels, mapped_labels = jnp.unique(labels, return_inverse=True)
    images_train, label_train, images_test, label_test = jax_train_test_split(images, mapped_labels, test_fraction=split, seed=seed)
    return images_train, label_train, images_test, label_test, mapped_labels, n_chunks

def loss_fun(model: nnx.Module, data: jax.Array, labels: jax.Array):
    logits = model(data)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    return loss, logits

@nnx.jit  # JIT-compile the function
def train_step( model: nnx.Module, optimizer: nnx.Optimizer, data: jax.Array, labels: jax.Array):
    loss_gradient = nnx.grad(loss_fun, has_aux=True)  # gradient transform!
    grads, logits = loss_gradient(model, data, labels)
    optimizer.update(grads)  # inplace update

def jax_train_test_split(features, labels, test_fraction=0.25, seed=0):
    features = jnp.asarray(features)
    labels = jnp.asarray(labels)

    n_samples = features.shape[0]
    key = jax.random.key(seed)

    # shuffle indices
    indices = jax.random.permutation(key, n_samples)
    features = features[indices]
    labels = labels[indices]

    # Split
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_test

    features_train = features[:n_train]
    labels_train = labels[:n_train]

    features_test = features[n_train:]
    labels_test = labels[n_train:]

    return features_train, labels_train, features_test, labels_test

class SimpleNN(nnx.Module):
    def __init__(self, n_features: int = 19200, n_hidden: int = 255, n_targets: int = 26, *, rngs: nnx.Rngs):
        self.n_features = n_features
        self.layer0 = nnx.Linear(n_features, n_features, rngs=rngs)
        #self.layer00 = nnx.Linear(n_features, int(n_features/3), rngs=rngs)
        self.layer1 = nnx.Linear(int(n_features), n_hidden, rngs=rngs)
        self.layer2 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer3 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer4 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer5 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer6 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.output = nnx.Linear(n_hidden, n_targets, rngs=rngs)

    def __call__(self, x):
        #x = x.reshape(x.shape[0], self.n_features) # Flatten images.
        x = nnx.selu(self.layer0(x))
        #x = nnx.selu(self.layer00(x))
        x = nnx.selu(self.layer1(x))
        x = nnx.selu(self.layer2(x))
        x = nnx.selu(self.layer3(x))
        x = nnx.selu(self.layer4(x))
        x = nnx.selu(self.layer5(x))
        x = nnx.selu(self.layer6(x))
        x = self.output(x)
        #x = self.layer3(x)
        return x

chunk_size = 126
#noise = 0.3
#l_r=0.06

images_train, label_train, images_test, label_test, mapped_labels, n_chunks = lode_data()
key = jax.random.key(int(time.time()))


#for noise in range(0.01, 0.04, 0.005):
#    for l_r in range(0.0, 0.8, 0.05):

for noise in jnp.arange(0.01, 0.05, 0.0075):
    for l_r in jnp.arange(0.0, 0.1, 0.0075):

        model = SimpleNN(n_features = n_chunks * (chunk_size // 2 + 1), rngs=nnx.Rngs(0))
        nnx.display(model)  # Interactive display if penzai is installed.
        optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=l_r), wrt=nnx.Param, )

        loserade = []
        for i in range(40):  # 300 training epochs
            key, k = jax.random.split(key)
            noisy_images_train = images_train + jax.random.normal(k, shape=images_train.shape, dtype=images_train.dtype) * noise
            train_step(model, optimizer, noisy_images_train , label_train)
            if i % 2 == 0:  # Print metrics.
                loss, _ = loss_fun(model, images_test, label_test)
                loserade.append(loss)
        with open(f"t/12/out_put_noise_{noise}_ln_{l_r}.txt", "w") as file:
            print(f"noise\t{noise}\tln\t{l_r}", file=file)
            print(f"noise\t{noise}\tln\t{l_r}")
            for lr in loserade:
                print(lr, file=file)

