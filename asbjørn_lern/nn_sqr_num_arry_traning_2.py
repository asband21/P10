import sys
import jax
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp

def test_funktion(x):
    return x*0.2 - 0.5

#def test_funktion(x):
#    of = 1.0
#    return jnp.where(x < 0, x * 0.5 + of, -0.1 * x + of)

#def test_funktion(x):
#    return jnp.sin(x)*0.5+0.5

def error(estimated, true):
    return jax.numpy.power((true - estimated), 2)

def neuron(a, b, x):
    return x*a + b

def net_make(n , relu_bool):
    rl = relu_bool
    s = " neuron( a[0], b[0], x)"
    for i in range(1,n):
        if rl:
            s = f"jax.nn.leaky_relu({s})"
        s = f" neuron( a[{i}], b[{i}], {s})"
    s = "lambda a, b, x :" + s
    f = eval(s)
    return f

def error_net_make(n, relu_bool):
    rl = relu_bool
    s = " neuron( a[0], b[0], x)"
    for i in range(1,n):
        if rl:
            s = f"jax.nn.leaky_relu({s})"
        s = f" neuron( a[{i}], b[{i}], {s})"
    s = "lambda a, b, x, true: error(" + s + ", true)"
    f = eval(s)
    return f

def traning(i, carry):
    a, b, ind_put, true, lerning_rate, error_over_time, net, error_gradine_funksen_a, error_gradine_funksen_b = carry

    am = a - lerning_rate * error_gradine_funksen_a(a, b, ind_put[i], true[i])
    bm = b - lerning_rate * error_gradine_funksen_b(a, b, ind_put[i], true[i])
    a = am
    b = bm

    return (a, b, ind_put, true, lerning_rate, error_over_time, net, error_gradine_funksen_a, error_gradine_funksen_b)

def traning_3(i, carry):
    a, b, ind_put, true, lerning_rate, bash = carry

    am = jnp.zeros_like(a)
    bm = jnp.zeros_like(b)

    foset = i*bash
    par_l = a, b, am, bm, ind_put, true, foset
    val_l = jax.lax.fori_loop(0, bash, bash_tren, par_l)
    a, b, am, bm, ind_put, true, foset = val_l

    a = a - am*lerning_rate
    b = b - bm*lerning_rate

    return (a, b, ind_put, true, lerning_rate, bash)

def bash_tren(i ,carry):
    a, b, am, bm, ind_put, true, foset = carry
    am = am + error_gradine_funksen_a(a, b, ind_put[i + foset], true[i + foset])
    bm = bm + error_gradine_funksen_b(a, b, ind_put[i + foset], true[i + foset])
    return (a, b, am, bm, ind_put, true, foset)


def traning_2(i, carry):
    a, b, ind_put, true, lerning_rate, bash = carry

    am = a - lerning_rate * error_gradine_funksen_a(a, b, ind_put[i], true[i])
    bm = b - lerning_rate * error_gradine_funksen_b(a, b, ind_put[i], true[i])
    a = am
    b = bm

    return (a, b, ind_put, true, lerning_rate, bash)


bool_relu = True
n = 6 
neurl_net_funksen = net_make(n ,bool_relu)
error_neurl_net_funksen = error_net_make(n ,bool_relu)

## make network veates 
key = jax.random.key(int(time.time()))
key, k1 = jax.random.split(key)
a = jax.random.uniform(k1, shape=(n), minval=-1.0, maxval=1.0)

key, k2 = jax.random.split(key)
b = jax.random.uniform(k2, shape=(n), minval=-1.0, maxval=1.0)

## run test net 
print(f"neurl_net_funksen(a, b, 33)={neurl_net_funksen(a, b, 33)}")

## gradient array test
gradine_funksen_a = jax.grad(neurl_net_funksen, 0)
gradine_funksen_b = jax.grad(neurl_net_funksen, 1)

ga = gradine_funksen_a(a, b, 33)
gb = gradine_funksen_b(a, b, 33)

print(f"a:{ga}")
print(f"b:{gb}")

error_gradine_funksen_a = jax.grad(error_neurl_net_funksen, 0)
error_gradine_funksen_b = jax.grad(error_neurl_net_funksen, 1)

ind_put = 5
true = test_funktion(ind_put)

ga = error_gradine_funksen_a(a, b, ind_put, true)
gb = error_gradine_funksen_b(a, b, ind_put, true)

print(f"error a:{ga}")
print(f"error b:{gb}")

## traning setup

bash = 10
traning_run = 10000
lerning_rate = 0.001

traning_rounds = traning_run*bash

key, k3 = jax.random.split(key)
ind_put = jax.random.uniform(k3, shape=(traning_rounds), minval=-20.0, maxval=20.0)
true    = test_funktion(ind_put)
error_over_time = jnp.zeros(traning_rounds)


print(f"a:{a}")
## traning loop
a_g = a 
par = a, b, ind_put, true, lerning_rate, bash
val = jax.lax.fori_loop(0, traning_run, traning_3, par)
a, b, ind_put, true, lerning_rate, bash = val
print(f"a     :{a}")
print(f"a diff:{a - a_g}")



t = jnp.arange(-20, 20, 0.05)
#t = jnp.arange(-20, 20, 1)
true_data = test_funktion(t)
estimated_data = jax.vmap(neurl_net_funksen, in_axes=(None, None, 0))(a, b, t)

plt.figure()
plt.plot(t, estimated_data, label="estimated")
plt.plot(t, true_data, label=" true")
plt.xlabel("x")
plt.ylabel("y")
plt.title("test fit")
plt.show()






