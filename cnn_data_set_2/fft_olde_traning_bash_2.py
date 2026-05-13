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

#jax.config.update('jax_platform_name', 'cpu')

def lode_data(split: float = 0.1, min_fri: float = 20000, seed: int = 5212, batch_size: int = 64):
    with open("./data_set/data_set_2_shuffel.csv", "r") as f:
        rows = [line.strip().split("\t") for line in f]

    
    si = int(len(rows)*split) #split_index
    rows_test = rows[:si]
    rows_train = rows[si:]

    old_labels = [5, 7, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 60]
    new_labels = [0, 1, 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    
    images_train = []
    images_test  = []
    
    labels_train = []
    labels_test  = []
    d = 1
    for bash_num in range(0, int(len(rows_test)/d), batch_size):
        batch_rows = rows_test[bash_num:bash_num + batch_size]
        images = []
        labels  = []

        for i in batch_rows:
            #if i[0] == 0:
            #    continue
            print(i)
            sr, x = wavfile.read("data_set/"+i[3])
            chunk_size = math.ceil(sr / min_fri)  
            x = x.astype("float32")
            x = x / 32768.0
            
            n_chunks = len(x) // chunk_size
            x = x[:n_chunks * chunk_size]
            x_spl = jnp.array(x).reshape(n_chunks, chunk_size)
            #x_fft = jnp.abs(jnp.fft.rfft(x_spl, axis=-1)) #.flatten()
            x_fft = jnp.abs(jnp.fft.rfft(x_spl, axis=-1)).flatten()

            if len(x_fft) != 41668:
                print(i)
                continue
            
            images.append(x_fft)
            labels.append(int(i[0]))

        #print(jnp.shape(images))
        images = jnp.asarray(images, dtype=jnp.float32)
        images_test.append(images)
        
        labels = apply_label_mapping(labels, old_labels, new_labels)
        labels = jnp.asarray(labels, dtype=jnp.int32)
        labels_test.append(labels)

    for bash_num in range(0, int(len(rows_train)/d), batch_size):
        batch_rows = rows_train[bash_num:bash_num + batch_size]
        images = []
        labels  = []

        for i in batch_rows:
            sr, x = wavfile.read("data_set/"+i[3])
            x = x.astype("float32")
            x = x / 32768.0
            
            n_chunks = len(x) // chunk_size
            x = x[:n_chunks * chunk_size]
            x_spl = jnp.array(x).reshape(n_chunks, chunk_size)
            x_fft = jnp.abs(jnp.fft.rfft(x_spl, axis=-1)).flatten()
            
            if len(x_fft) != 41668:
                print(i)
                continue

            images.append(x_fft)
            labels.append(int(i[0]))
            
        #print(jnp.shape(images))
        images = jnp.asarray(images, dtype=jnp.float32)
        images_train.append(images)
        
        labels = apply_label_mapping(labels, old_labels, new_labels)
        labels = jnp.asarray(labels, dtype=jnp.int32)
        labels_train.append(labels)

    #unique_labels, mapped_labels = jnp.unique(labels_train, return_inverse=True)

    return images_train, labels_train, images_test, labels_test, n_chunks

def apply_label_mapping(labels, old_labels, new_labels):
    label_map = dict(zip(old_labels, new_labels))
    return [label_map[label] for label in labels]

def loss_fun(model: nnx.Module, data: jax.Array, labels: jax.Array):
    logits = model(data)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    #loss = jax.numpy.power((labels - logits), 2).mean()
    return loss, logits

@nnx.jit  # JIT-compile the function
def train_step( model: nnx.Module, optimizer: nnx.Optimizer, data: jax.Array, labels: jax.Array):
    loss_gradient = nnx.grad(loss_fun, has_aux=True)  # gradient transform!
    grads, logits = loss_gradient(model, data, labels)
    #optimizer.update(model, grads)
    optimizer.update(grads)  # inplace update

class SimpleNN(nnx.Module):
    def __init__(self, n_features: int = 19200, n_hidden: int = 255, n_targets: int = 26, *, rngs: nnx.Rngs):
        self.n_features = n_features
        #self.layer0 = nnx.Linear(n_features, int(n_features/4), rngs=rngs)
        #self.layer00 = nnx.Linear(n_features, int(n_features/3), rngs=rngs)
        #self.layer1 = nnx.Linear(int(n_features/4), n_hidden, rngs=rngs)
        self.layer1 = nnx.Linear(n_features, n_hidden, rngs=rngs)
        self.layer2 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer3 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer4 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        #self.layer5 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        #self.layer6 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.output = nnx.Linear(n_hidden, n_targets, rngs=rngs)

    def __call__(self, x):
        #x = x.reshape(x.shape[0], self.n_features) # Flatten images.
        #x = nnx.relu6(self.layer0(x))
        #x = nnx.selu(self.layer00(x))
        x = nnx.relu6(self.layer1(x))
        x = nnx.relu6(self.layer2(x))
        x = nnx.relu6(self.layer3(x))
        x = nnx.relu6(self.layer4(x))
        #x = nnx.relu6(self.layer5(x))
        #x = nnx.relu6(self.layer6(x))
        x = self.output(x)
        #x = self.layer3(x)
        return x

noise = 0.01
l_r=0.0000002
#l_r=0.0000004

batch_size = 32
images_train, labels_train, images_test, labels_test, n_chunks = lode_data(batch_size = batch_size)

n_f = 22656
#n_f = 44604
#n_f = 22506
n_f = 41668
n_t = 19
model = SimpleNN(n_features = n_f, n_targets = n_t, rngs=nnx.Rngs(0), n_hidden = 2048)
nnx.display(model)  # Interactive display if penzai is installed.

print(f"{model(images_train[0][0])} = model({images_train[0][0]})")


run_id = int(time.time())
run_dir = Path.cwd() / "model_fft" / str(run_id)
os.makedirs(run_dir)

checkpointer = ocp.StandardCheckpointer()
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=l_r), wrt=nnx.Param, )

key = jax.random.key(int(time.time()))
loserade = []
for epoch in range(256*4):
    #for bash_i in : 
    #print(images_train)
    #print(len(images_train))
    for im_tr in range(len(images_train)):
        #key, k = jax.random.split(key)
        #noisy_images_train = images_train + jax.random.normal(k, shape=images_train.shape, dtype=images_train.dtype) * noise
        train_step(model, optimizer, images_train[im_tr], labels_train[im_tr])
    
    #if epoch % 5 == 0:  # Print metrics.
    #    key, k = jax.random.split(key)
    #    r_i = jax.random.randint(k, shape=(), minval=0, maxval=len(images_test))
    #    loss, _ = loss_fun(model, images_test[r_i], labels_test[r_i])
    #    loserade.append(loss)
    #    print(f"epoch\t{epoch}\tloss\t{loss}")
    
    if epoch % 5 == 0:  # Print metrics.
        loss_sum = 0
        for im_tr in range(len(images_test)):
            loss, _ = loss_fun(model, images_test[im_tr], labels_test[im_tr])
            loss_sum = loss_sum + loss
        loserade.append(loss_sum/len(images_test))
        print(f"epoch\t{epoch}\tloss\t{loss_sum}")


_, state = nnx.split(model)
run_dir.mkdir(parents=True, exist_ok=True)
checkpointer.save(run_dir / "state", state)
checkpointer.wait_until_finished()

print(f"Saved checkpoint to: {run_dir}")

print(f"\t{l_r}")
for i in loserade:
    print(i)
