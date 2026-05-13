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
        if 500 < num:
            break

        sr, x = wavfile.read(i[2])
        x = x.astype("float32")
        x = x / 32768.0
        if spl_amt != 0:
            x = x[:len(x) // spl_amt]
        n_chunks = len(x) // chunk_size
        x = x[:n_chunks * chunk_size]
        x_spl_0 = jnp.array(x[:,0]).reshape(n_chunks, chunk_size)
        x_spl_1 = jnp.array(x[:,1]).reshape(n_chunks, chunk_size)
        
        x_fft_0 = jnp.log1p(jnp.abs(jnp.fft.rfft(x_spl_0, axis=-1))).flatten()
        x_fft_1 = jnp.log1p(jnp.abs(jnp.fft.rfft(x_spl_1, axis=-1))).flatten()

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
    labels = jnp.asarray(labels, dtype=jnp.float32)
    distens = jnp.asarray(distens, dtype=jnp.float32)
    unique_labels, mapped_labels = jnp.unique(labels, return_inverse=True)
    images_train, label_train, images_test, label_test = jax_train_test_split(images, distens, test_fraction=split, seed=seed)
    return images_train, label_train, images_test, label_test, mapped_labels, n_chunks

def loss_fun(model: nnx.Module, data: jax.Array, labels: jax.Array):
    logits = model(data)
    #loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    loss = jax.numpy.power((labels - logits), 2).mean()
    return loss, logits


@nnx.jit  # JIT-compile the function
def train_step( model: nnx.Module, optimizer: nnx.Optimizer, data: jax.Array, labels: jax.Array):
    loss_gradient = nnx.grad(loss_fun, has_aux=True)  # gradient transform!
    grads, logits = loss_gradient(model, data, labels)
    #optimizer.update(model, grads)
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
        self.layer0 = nnx.Linear(n_features, int(n_features/8), rngs=rngs)
        #self.layer00 = nnx.Linear(n_features, int(n_features/3), rngs=rngs)
        self.layer1 = nnx.Linear(int(n_features/8), n_hidden, rngs=rngs)
        self.layer2 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer3 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        #self.layer4 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        #self.layer5 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        #self.layer6 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.output = nnx.Linear(n_hidden, n_targets, rngs=rngs)

    def __call__(self, x):
        #x = x.reshape(x.shape[0], self.n_features) # Flatten images.
        x = nnx.relu6(self.layer0(x))
        #x = nnx.selu(self.layer00(x))
        x = nnx.relu6(self.layer1(x))
        x = nnx.relu6(self.layer2(x))
        x = nnx.relu6(self.layer3(x))
        #x = nnx.relu6(self.layer4(x))
        #x = nnx.relu6(self.layer5(x))
        #x = nnx.relu6(self.layer6(x))
        x = self.output(x)
        #x = self.layer3(x)
        return x

chunk_size = 64
noise = 0.01
l_r=0.01

images_train, label_train, images_test, label_test, mapped_labels, n_chunks = lode_data(spl_amt=0)

audio_chanel = 1
fit = audio_chanel*n_chunks * (chunk_size // 2 + 1)
sha = jnp.shape(label_train) 
model = SimpleNN(n_features = fit, n_targets = sha[1] , n_hidden = 1024, rngs=nnx.Rngs(0))

nnx.display(model)  # Interactive display if penzai is installed.

run_id = int(time.time())
run_dir = Path.cwd() / "model_fft" / str(run_id)
os.makedirs(run_dir)

checkpointer = ocp.StandardCheckpointer()
optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=l_r), wrt=nnx.Param, )

#key = jax.random.key(int(time.time()))

loserade = []
#for i in range(300):  # 300 training epochs
    #key, k = jax.random.split(key)
    #noisy_images_train = images_train + jax.random.normal(k, shape=images_train.shape, dtype=images_train.dtype) * noise
#    train_step(model, optimizer, images_train , label_train)
#    if i % 5 == 0:  # Print metrics.
#        loss, _ = loss_fun(model, images_test, label_test)
#        loserade.append(loss)
#        print(f"epoch\t{i}\tloss\t{loss}")

batch_size = 8

for epoch in range(80):
    for start in range(0, images_train.shape[0], batch_size):
        end = start + batch_size
        print(start)
        x_batch = images_train[start:end]
        y_batch = label_train[start:end]

        train_step(model, optimizer, x_batch, y_batch)
        #train_step(model, optimizer, images_train , label_train)

    if epoch % 2 == 0:
        loss, _ = loss_fun(model, images_test, label_test)
        loss_value = float(loss)
        loserade.append(loss_value)
        print(f"epoch\t{epoch}\tloss\t{loss_value}")


_, state = nnx.split(model)
run_dir.mkdir(parents=True, exist_ok=True)
checkpointer.save(run_dir / "state", state)
checkpointer.wait_until_finished()

print(f"Saved checkpoint to: {run_dir}")

print(f"\t{l_r}")
for i in loserade:
    print(i)
