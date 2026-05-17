import os       
import jax
import math
import time
import json
import optax
import argparse
from flax import nnx
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from scipy.io import wavfile
import orbax.checkpoint as ocp

#jax.config.update('jax_platform_name', 'cpu')

def lode_data(split: float = 0.1, min_fri: float = 20000, seed: int = 5212, batch_size: int = 64, d=1):
    #with open("./data_set/data_set_2_shuffel.csv", "r") as f:
    with open("./data_set/data_set_clean.csv", "r") as f:
        rows = [line.strip().split("\t") for line in f]
    for lolo in rows:
        lolo[3] = "data_set/" + lolo[3]

    with open("./data_set_2/data_set_clean.csv", "r") as f_2:
        rows_2 = [line.strip().split("\t") for line in f_2]
    for lol in rows_2:
        lol[3] = "data_set_2/" + lol[3]
    rows.extend(rows_2)

    key = jax.random.key(seed)
    shuffled_indices = jax.random.permutation(key, len(rows))
    rows_shuffled = []
    for lll in shuffled_indices:
        rows_shuffled.append(rows[lll])

    si = int(len(rows)*split) #split_index
    rows_test = rows_shuffled[:si]
    rows_train = rows_shuffled[si:]

    old_labels = [5, 7, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]
    new_labels = [0, 1, 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
    images_train = []
    images_test  = []
    
    labels_train = []
    labels_test  = []
    #d = 1
    for bash_num in range(0, int(len(rows_test)/d), batch_size):
        print(f"\rlode test {bash_num}/{len(rows_test)}", end='')
        batch_rows = rows_test[bash_num:bash_num + batch_size]
        images = []
        labels  = []

        for i in batch_rows:
            #if i[0] == 0:
            #    continue
            #print(i)
            sr, x = wavfile.read(i[3])
            if len(x) !=75776:
                print(i)
                continue
            
            chunk_size = math.ceil(sr / min_fri)  
            x = x.astype("float32")
            x = x / 32768.0
            
            n_chunks = len(x) // chunk_size
            x = x[:n_chunks * chunk_size]
            x_spl = np.array(x).reshape(n_chunks, chunk_size)
            #x_fft = jnp.abs(jnp.fft.rfft(x_spl, axis=-1)) #.flatten()
            x_fft = np.abs(np.fft.rfft(x_spl, axis=-1)) #.flatten()

            
            images.append(x_fft)
            labels.append(int(i[0]))

        #print(jnp.shape(images))
        images = np.asarray(images, dtype=np.float32)
        images = images[..., None]
        images_test.append(images)
        
        labels = apply_label_mapping(labels, old_labels, new_labels)
        labels = np.asarray(labels, dtype=np.int32)
        labels_test.append(labels)
    print()
    for bash_num in range(0, int(len(rows_train)/d), batch_size):
        print(f"\rlode train {bash_num}/{len(rows_train)}", end='')
        batch_rows = rows_train[bash_num:bash_num + batch_size]
        images = []
        labels  = []

        for i in batch_rows:
            sr, x = wavfile.read(i[3])
            if len(x) != 75776:
                print()
                print(i)
                print(len(x))
                print(jnp.shape(x))
                continue
            x = x.astype("float32")
            x = x / 32768.0
            
            n_chunks = len(x) // chunk_size
            x = x[:n_chunks * chunk_size]
            x_spl = np.array(x).reshape(n_chunks, chunk_size)
            x_fft = np.abs(jnp.fft.rfft(x_spl, axis=-1)) #.flatten()
            

            images.append(x_fft)
            labels.append(int(i[0]))
            
        #print(jnp.shape(images))
        images = np.asarray(images, dtype=np.float32)
        images = images[..., None]
        images_train.append(images)
        
        labels = apply_label_mapping(labels, old_labels, new_labels)
        labels = jnp.asarray(labels, dtype=jnp.int32)
        labels_train.append(labels)

    return images_train, labels_train, images_test, labels_test, n_chunks

def parse_args():
    parser = argparse.ArgumentParser(description="Train FFT model on audio dataset.")
    
    parser.add_argument("--noise", type=float, default=0.0002)
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.0000002)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=256 * 2)
    parser.add_argument("--minimum-fri", type=float, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-features", type=int, default=41668)
    parser.add_argument("--n-targets", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--data-frak", type=int, default=1)
    parser.add_argument("--model", type=str, default=None)

    return parser.parse_args()

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

class cnn(nnx.Module):
    def __init__(self, n_in_len: int = 300, n_in_hite: int = 64, n_in_deeb: int = 1, n_hidden: int = 255, n_targets: int = 20, kn_size: int = 5,  *, rngs: nnx.Rngs):
        self.layer1 = nnx.Conv(in_features=n_in_deeb, out_features=5, kernel_size=(5,5),  padding='VALID', rngs=rngs)
        self.layer2 = nnx.Conv(in_features=5, out_features=5, kernel_size=(5,5),  padding='VALID', rngs=rngs)
        self.layer3 = nnx.Conv(in_features=5, out_features=5, kernel_size=(5,5),  padding='VALID', rngs=rngs)
        self.layer4 = nnx.Conv(in_features=5, out_features=5, kernel_size=(5,5),  padding='VALID', rngs=rngs)
        self.layer5 = nnx.Conv(in_features=5, out_features=1, kernel_size=(5,5),  padding='VALID', rngs=rngs)
        #self.layer6 = nnx.Linear(n_in_len*n_in_hite, n_hidden, rngs=rngs)
        self.layer6 = nnx.Linear(28798, n_hidden, rngs=rngs)
        self.layer7 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer8 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.layer9 = nnx.Linear(n_hidden, n_hidden, rngs=rngs)
        self.output = nnx.Linear(n_hidden, n_targets, rngs=rngs)

    def __call__(self, x):
        #if x.ndim == 3:
            #x = x[None, ...]

        #print(jnp.shape(x))
        x = nnx.relu6(self.layer1(x))
        #print(jnp.shape(x))
        x = nnx.relu6(self.layer2(x))
        #print(jnp.shape(x))
        x = nnx.relu6(self.layer3(x))
        #print(jnp.shape(x))
        x = nnx.relu6(self.layer4(x))
        #print(jnp.shape(x))
        x = nnx.relu6(self.layer5(x))
        #print(jnp.shape(x))
        #x = x.flatten()
        x = x.reshape((x.shape[0], -1))
        #print(jnp.shape(x))
        x = nnx.relu6(self.layer6(x))
        x = nnx.relu6(self.layer7(x))
        x = nnx.relu6(self.layer8(x))
        x = nnx.relu6(self.layer9(x))
        x = self.output(x)
        #x = nnx.sigmoid(self.output(x))
        return x

args = parse_args()

noise = args.noise
l_r = args.learning_rate
batch_size = args.batch_size
epoch = args.epochs
minimum_fri = args.minimum_fri
seed = args.seed if args.seed is not None else int(time.time())
n_f = args.n_features
n_t = args.n_targets
n_hidden = args.hidden_size
data_frak = args.data_frak


images_train, labels_train, images_test, labels_test, n_chunks = lode_data(min_fri=minimum_fri, batch_size = batch_size, d=data_frak) #data_frak)
#model = SimpleNN(n_features = n_f, n_targets = n_t, rngs=nnx.Rngs(0), n_hidden = 2048)
shape = jnp.shape(images_train[0][0])
print(shape)
model = cnn(n_in_len = shape[1], n_in_hite = shape[2], n_targets=20, rngs=nnx.Rngs(0))


nnx.display(model)  # Interactive display if penzai is installed.
print(f"noise {noise} learning rate {l_r} batch_size {batch_size} epoch {epoch}k seed {seed} n_f {n_f} n_t {n_t} minimum_fri {minimum_fri}")
x = images_train[0][0][None, ...]
print(f"{model(x)} = model({images_train[0][0]})")


run_id = int(time.time())
run_dir = Path.cwd() / "model_fft" / str(run_id)
#os.makedirs(run_dir)
run_dir.mkdir(parents=True, exist_ok=True)

checkpointer = ocp.StandardCheckpointer()
if args.model != None:
    #_, state = nnx.split(model)
    checkpoint_path = Path(args.model).resolve()
    _, state = nnx.split(model)
    state = checkpointer.restore(checkpoint_path, state, partial_restore=True)
    nnx.update(model, state)

optimizer = nnx.Optimizer(model, optax.adam(learning_rate=l_r), wrt=nnx.Param, )


key = jax.random.key(seed)
loserade = []
for epoch in range(epoch):
    # train model 
    key, k = jax.random.split(key)
    shuffled_indices = jax.random.permutation(k, len(images_train))
    for im_tr in range(len(images_train)):
        key, k = jax.random.split(key)
        idx = int(shuffled_indices[im_tr])
        noisy_images_train = jnp.array(images_train[idx]) + jax.random.normal(k, shape=images_train[idx].shape, dtype=images_train[idx].dtype) * noise
        train_step(model, optimizer, noisy_images_train, labels_train[idx])
    
    if epoch % 5 == 0:  # print loss.
        loss_sum = 0
        for im_tr in range(len(images_test)):
            i_t = jnp.array(images_test[im_tr])
            loss, _ = loss_fun(model, i_t, labels_test[im_tr])
            loss_sum = loss_sum + loss
        loserade.append(loss_sum/len(images_test))
        print(f"epoch\t{epoch}\tloss\t{loss_sum/len(images_test)}")

    #if epoch in {100, 200, 300, 400}: # save model
    if epoch in {200, 400, 600, 800}: # save model
        _, state = nnx.split(model)
        state_cpu = jax.device_get(state)
        ckpt_path = run_dir / f"state_{epoch}"
        checkpointer.save(ckpt_path, state_cpu)
        checkpointer.wait_until_finished()
        del state_cpu
        del state
        jax.clear_caches()
        print(f"epoch {epoch} saved checkpoint to: {ckpt_path}")

_, state = nnx.split(model)
state_cpu = jax.device_get(state)
checkpointer.save(run_dir / "state_last", state)
checkpointer.wait_until_finished()
del state_cpu
print(f"Saved checkpoint to: {run_dir}")

for i in loserade:
    print(i)
