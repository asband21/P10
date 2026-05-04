import jax
import optax
from flax import nnx
import jax.numpy as jnp
from scipy.io import wavfile

def lode_data(split: float = 0.2):
    with open("../data_set/data_set_3.csv", "r") as f:
        rows = [line.strip().split("\t") for line in f]
    images = []
    labels  = []
    for i in range(3,1001,1):
        sr, x = wavfile.read("../data_set/" + rows[i][3])
        x = x.astype("float32")
        x = x / 32768.0
        images.append(x)
        labels.append(int(rows[i][0]))
        #print(len(x))
    images = jnp.asarray(images, dtype=jnp.float32)
    labels = jnp.asarray(labels, dtype=jnp.int32)
    unique_labels, mapped_labels = jnp.unique(labels, return_inverse=True)
    #images = jnp.array(images)
    #labels = jnp.array(labels)
    #images_train, label_train, images_test, label_test = jax_train_test_split(images, unique_labels, test_fraction=split, seed=0)
    images_train, label_train, images_test, label_test = jax_train_test_split(images, mapped_labels, test_fraction=split, seed=0)
    return images_train, label_train, images_test, label_test, mapped_labels
    #return images_train, label_train

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

    # Native JAX shuffle indices
    indices = jax.random.permutation(key, n_samples)

    # Shuffle features and labels in the same way
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
    #def __init__(self, n_features: int = 113664, n_hidden_h: int = 113664, n_hidden_l: int = 3, n_targets: int = 26, *, rngs: nnx.Rngs):
    def __init__(self, n_features: int = 113664, n_hidden_h: int = 100, n_hidden_l: int = 100, n_targets: int = 26, *, rngs: nnx.Rngs):
        self.n_features = n_features
        self.layer1 = nnx.Linear(n_features, n_hidden_h, rngs=rngs)
        self.layer2 = nnx.Linear(n_hidden_h, n_hidden_l, rngs=rngs)
        self.layer3 = nnx.Linear(n_hidden_h, n_targets, rngs=rngs)

    def __call__(self, x):
        x = x.reshape(x.shape[0], self.n_features) # Flatten images.
        x = nnx.selu(self.layer1(x))
        x = nnx.selu(self.layer2(x))
        x = self.layer3(x)
        return x

model = SimpleNN(rngs=nnx.Rngs(0))

#nnx.display(model)  # Interactive display if penzai is installed.

optimizer = nnx.ModelAndOptimizer(model, optax.sgd(learning_rate=0.05))

images_train, label_train, images_test, label_test, mapped_labels = lode_data()
#print(label_train)
#print(mapped_labels)

for i in range(1000):  # 300 training epochs
    train_step(model, optimizer, images_train, label_train)
    #print(i)
    if i % 20 == 0:  # Print metrics.
        loss, _ = loss_fun(model, images_test, label_test)
        print(f"epoch {i}: loss={loss:.2f}")
