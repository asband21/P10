import sys
import jax
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp

def test_funktion(x):
    return 0.1*x*x + x*1.8 - 0.5

def Relu(x):
    return jnp.maximum(x, 0)

def error(estimated, true):
    return jax.numpy.power((true - estimated), 2)

def neuron(a, b, x):
    return x*a + b

def net(a, b, a2, b2, a3, b3, x, true):
    x = net_true(a, b, a2, b2, a3, b3, x)
    return error(x, true)

def net_true(a, b, a2, b2, a3, b3, x):
    x = neuron(a, b, x)
    x = jax.nn.relu(x)
    x = neuron(a2, b2, x)
    x = jax.nn.relu(x)
    x = neuron(a3, b3, x)
    #x = jax.nn.relu(x)
    return x

def traning(i, carry):
    a, b, a2, b2, a3, b3, fun_input, true, error_over_time, l = carry
    
    a  = a  - l*grad_network_a_error( a, b, a2, b2, a3, b3, fun_input[i], true[i])
    b  = b  - l*grad_network_b_error( a, b, a2, b2, a3, b3, fun_input[i], true[i])
    a2 = a2 - l*grad_network_a2_error(a, b, a2, b2, a3, b3, fun_input[i], true[i])
    b2 = b2 - l*grad_network_b2_error(a, b, a2, b2, a3, b3, fun_input[i], true[i])
    a3 = a3 - l*grad_network_a3_error(a, b, a2, b2, a3, b3, fun_input[i], true[i])
    b3 = b3 - l*grad_network_b3_error(a, b, a2, b2, a3, b3, fun_input[i], true[i])
    
    #jax.debug.print("a:{} b:{} a_m:{} b_m:{}", a, b, a_m, b_m)
    error_over_time = error_over_time.at[i].set(net(a, b, a2, b2, a3, b3, fun_input[i], true[i]))
    return (a, b, a2, b2, a3, b3, fun_input, true, error_over_time, l)


key = jax.random.key(int(time.time()))
#key = jax.random.key(10)
key, k1 = jax.random.split(key)
a = jax.random.uniform(k1, shape=(), minval=-1.0, maxval=1.0)

key, k2 = jax.random.split(key)
b = jax.random.uniform(k2, shape=(), minval=-1.0, maxval=1.0)

key, k3 = jax.random.split(key)
a2 = jax.random.uniform(k3, shape=(), minval=-1.0, maxval=1.0)

key, k4 = jax.random.split(key)
b2 = jax.random.uniform(k4, shape=(), minval=-1.0, maxval=1.0)

key, k5 = jax.random.split(key)
a3 = jax.random.uniform(k5, shape=(), minval=-1.0, maxval=1.0)

key, k6 = jax.random.split(key)
b3 = jax.random.uniform(k6, shape=(), minval=-1.0, maxval=1.0)


## maked the gradient funksen 
grad_network_a_error = jax.grad(net, 0)
grad_network_b_error = jax.grad(net, 1)
grad_network_a2_error = jax.grad(net, 2)
grad_network_b2_error = jax.grad(net, 3)
grad_network_a3_error = jax.grad(net, 4)
grad_network_b3_error = jax.grad(net, 5)

l = 0.0001
num = 20000

key, k60 = jax.random.split(key)
fun_input = jax.random.uniform(k60, shape=(num), minval=-10.0, maxval=10.0)

data_set = test_funktion(fun_input)
error_over_time = jnp.zeros(num)

## traning loop
lkl = (a, b, a2, b2, a3, b3, fun_input, data_set, error_over_time, l)
val = jax.lax.fori_loop(0, num, traning, lkl) 
a, b, a2, b2, a3, b3, sam, true, error_over_time, l = val


jax.debug.print("a:{} b:{} a2:{} b2:{} a3:{} b3:{}", a, b, a2, b2, a3, b3)

plt.figure()
plt.plot(error_over_time)
plt.title("Error plot")
plt.show()

t = jnp.arange(-20, 20, 0.05)
true_data = test_funktion(t)
estimated_data = net_true(a, b, a2, b2, a3, b3, t)

plt.figure()
plt.plot(t, estimated_data, label="estimated")
plt.plot(t, true_data, label=" true")
plt.xlabel("x")
plt.ylabel("y")
plt.title("test fit")
plt.show()



