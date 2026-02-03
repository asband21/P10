import sys
import jax
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp

def test_funktion(x):
    return jnp.cos(x)+1

def test_funktion(x):
    return jnp.cos(x*2)*jnp.sin(x)+1

def Relu(x):
    return jnp.maximum(x, 0.0)

def error(estimated, true):
    return jax.numpy.power((true - estimated), 2)
    #return jax.numpy.power((estimated - true), 2)

def neuron(a, b, x):
    return Relu(x*a + b)

#def neuron(a, b, x):
#    return Relu(x)*a + b

def network_1d(a1 ,a2 ,a3, a4, b1, b2, b3, b4 ,x):
    return neuron(a4 ,b4, neuron(a3 ,b3, neuron(a2 ,b2, neuron(a1 ,b1, x))))

#def network_1d(a1, a2, a3, a4, b1, b2, b3, b4, x):
#    x = neuron(a1, b1, x)
#    x = neuron(a2, b2, x)
#    x = neuron(a3, b3, x)
#    x = neuron(a4, b4, x)
#    return x

def network_1d_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4 ,x):
    return error(network_1d(a1 ,a2 ,a3, a4, b1, b2, b3, b4 ,x ), test_funktion(x))

def multigradiens(i, carry):
        a1_bash ,a2_bash ,a3_bash ,a4_bash ,b1_bash ,b2_bash ,b3_bash ,b4_bash, a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti, l= carry
        
        a1_bash = l*grad_network_a1_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)
        a2_bash = l*grad_network_a2_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)
        a3_bash = l*grad_network_a3_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)
        a4_bash = l*grad_network_a4_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)
        b1_bash = l*grad_network_a1_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)
        b2_bash = l*grad_network_b2_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)
        b3_bash = l*grad_network_b3_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)
        b4_bash = l*grad_network_b4_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)
        ti = ti + 0.01
        return (a1_bash ,a2_bash ,a3_bash ,a4_bash ,b1_bash ,b2_bash ,b3_bash ,b4_bash, a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti, l)

def print_wate(a1 ,a2 ,a3, a4, b1, b2, b3, b4):
    print("----network wate----")
    print("a1:" + str(a1))
    print("a2:" + str(a2))
    print("a3:" + str(a3))
    print("a4:" + str(a4))
    print("b1:" + str(b1))
    print("b2:" + str(b2))
    print("b3:" + str(b3))
    print("b4:" + str(a4))

def tranign_bash(i, carry):
    lering_rade, bash, a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti = carry
    a1_bash = 0
    a2_bash = 0
    a3_bash = 0
    a4_bash = 0
    b1_bash = 0
    b2_bash = 0
    b3_bash = 0
    b4_bash = 0
    val = jax.lax.fori_loop(0,    # lower
                      bash,      # upper
                      multigradiens, # body_fun
                      (a1_bash ,a2_bash ,a3_bash ,a4_bash ,b1_bash ,b2_bash ,b3_bash ,b4_bash, a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti, lering_rade)) 
    a1_bash ,a2_bash ,a3_bash ,a4_bash ,b1_bash ,b2_bash ,b3_bash ,b4_bash, a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti, lering_rade = val
    a1 = a1 + a1_bash
    a2 = a2 + a2_bash
    a3 = a3 + a3_bash
    a4 = a4 + a4_bash
    b1 = b1 + b1_bash
    b2 = b2 + b2_bash
    b3 = b3 + b3_bash
    b4 = b4 + b4_bash 
    ti = ti + 0.1
    return (lering_rade, bash, a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)

def traning(i, carry):
    nummer, bash, lering_rade, a1 ,a2 ,a3, a4, b1, b2, b3, b4 = carry
    ti = 1
    val = jax.lax.fori_loop(0, nummer, tranign_bash, (a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti, lering_rade)) 
    a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti, lering_rade = val
    return (bash, lering_rade, a1 ,a2 ,a3, a4, b1, b2, b3, b4)


## random nerorns wate

key = jax.random.key(int(time.time()))

a1 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
a2 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
a3 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
a4 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
b1 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
b2 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
b3 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
b4 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
print_wate(a1 ,a2 ,a3, a4, b1, b2, b3, b4)

#grad_network_a1_test = jax.grad(network_1d)
#print(grad_network_a1_test)
#grad_a1 = grad_network_a1_test(a1 ,a2 ,a3, a4, b1, b2, b3, b4 ,0.5)
#print(grad_a1)

## maked the gradient funksen 

grad_network_a1_error = jax.grad(network_1d_error, 0)
grad_network_a2_error = jax.grad(network_1d_error, 1)
grad_network_a3_error = jax.grad(network_1d_error, 2)
grad_network_a4_error = jax.grad(network_1d_error, 3)
grad_network_b1_error = jax.grad(network_1d_error, 4)
grad_network_b2_error = jax.grad(network_1d_error, 5)
grad_network_b3_error = jax.grad(network_1d_error, 6)
grad_network_b4_error = jax.grad(network_1d_error, 7)

"""
print("a1 grad at(" + str(ti) + "):" + str(grad_network_a1_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("a2 grad at(" + str(ti) + "):" + str(grad_network_a2_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("a3 grad at(" + str(ti) + "):" + str(grad_network_a3_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("a4 grad at(" + str(ti) + "):" + str(grad_network_a4_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("b1 grad at(" + str(ti) + "):" + str(grad_network_b1_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("b2 grad at(" + str(ti) + "):" + str(grad_network_b2_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("b3 grad at(" + str(ti) + "):" + str(grad_network_b3_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("b4 grad at(" + str(ti) + "):" + str(grad_network_b4_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
"""

## traning network 

ti = 7.5

val = jax.lax.fori_loop(0, 60, tranign_bash, (-0.01, 10, a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)) 
lering_rate, bash, a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti = val

print_wate(a1 ,a2 ,a3, a4, b1, b2, b3, b4)


## test network
t = jnp.arange(1.0, 50.0, 0.01)
true_data = test_funktion(t)
estimated_data = network_1d(a1 ,a2 ,a3, a4, b1, b2, b3, b4 ,t)
error_data = error(estimated_data, true_data)

plt.figure()
plt.plot(t, true_data, label="true")
plt.plot(t, estimated_data, label="estimated")
plt.plot(t, error_data, label="error")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Error plot")
plt.show()
