import sys
import jax
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp

def test_funktion(x):
    return x*0.8 - 0.5

def Relu(x):
    return jnp.maximum(x, 0)

def error(estimated, true):
    return jax.numpy.power((true - estimated), 2)

def neuron(a, b, x):
    return x*a + b

def net(a, b, x, true):
    return error(neuron(a, b, x), true)

key = jax.random.key(int(time.time()))
key = jax.random.key(10)
key, k1 = jax.random.split(key)
a = jax.random.uniform(k1, shape=(), minval=-1.0, maxval=1.0)

key, k5 = jax.random.split(key)
b = jax.random.uniform(k5, shape=(), minval=-1.0, maxval=1.0)

## maked the gradient funksen 

grad_network_a_error = jax.grad(net, 0)
grad_network_b_error = jax.grad(net, 1)

l = -0.001
num = 100

key, k6 = jax.random.split(key)
til = jax.random.uniform(k6, shape=(num), minval=-10.0, maxval=10.0)

data_set = test_funktion(til)

#print(til)
#print(data_set)
error_over_time = []
for i in range(num):
    a_m = l*grad_network_a_error(a,b, til[i], data_set[i])
    b_m = l*grad_network_b_error(a,b, til[i], data_set[i])
    a = a + a_m
    b = b + b_m
    
    #jax.debug.print("a:{} b:{} a_m:{} b_m:{}", a, b, a_m, b_m)
    error_over_time.append(net(a, b, til[i], data_set[i]))

plt.figure()
plt.plot(error_over_time)
plt.title("Error plot")
plt.show()
 
