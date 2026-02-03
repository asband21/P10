import sys
import jax
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp

def test_funktion(x):
    return jnp.cos(x)

#def Relu(x):
#    if 0 < x:
#        return x
#    return 0

def Relu(x):
    return jnp.maximum(x, 0.0)

def error(estimated, true):
    return jax.numpy.power((true - estimated), 2)

def neuron(a, b, x):
    return Relu(x*a + b)

def network_1d(a1 ,a2 ,a3, a4, b1, b2, b3, b4 ,x):
    return neuron(a4 ,b4, neuron(a3 ,b3, neuron(a2 ,b2, neuron(a1 ,b1, x))))

def network_1d_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4 ,x):
    return error(network_1d(a1 ,a2 ,a3, a4, b1, b2, b3, b4 ,x ), test_funktion(x))

def traning()

#def network_1d(a1, a2, a3, a4, b1, b2, b3, b4, x):
#    x = neuron(a1, b1, x)
#    x = neuron(a2, b2, x)
#    x = neuron(a3, b3, x)
#    x = neuron(a4, b4, x)
#    return x

key = jax.random.key(int(time.time()))
print(key)
a1 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
a2 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
a3 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
a4 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
b1 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
b2 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
b3 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)
b4 = jax.random.uniform(key, shape=(), dtype=None, minval=-1.0, maxval=1.0)

print(a1)

t = jnp.arange(1.0, 50.0, 0.01)


grad_network_a1_test = jax.grad(network_1d)
print(grad_network_a1_test)
grad_a1 = grad_network_a1_test(a1 ,a2 ,a3, a4, b1, b2, b3, b4 ,0.5)
print(grad_a1)

grad_network_a1_error = jax.grad(network_1d_error, 0)
grad_network_a2_error = jax.grad(network_1d_error, 1)
grad_network_a3_error = jax.grad(network_1d_error, 2)
grad_network_a4_error = jax.grad(network_1d_error, 3)
grad_network_b1_error = jax.grad(network_1d_error, 4)
grad_network_b2_error = jax.grad(network_1d_error, 5)
grad_network_b3_error = jax.grad(network_1d_error, 6)
grad_network_b4_error = jax.grad(network_1d_error, 7)

ti = 7.5
print("a1 grad at(" + str(ti) + "):" + str(grad_network_a1_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("a2 grad at(" + str(ti) + "):" + str(grad_network_a2_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("a3 grad at(" + str(ti) + "):" + str(grad_network_a3_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("a4 grad at(" + str(ti) + "):" + str(grad_network_a4_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("b1 grad at(" + str(ti) + "):" + str(grad_network_b1_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("b2 grad at(" + str(ti) + "):" + str(grad_network_b2_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("b3 grad at(" + str(ti) + "):" + str(grad_network_b3_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))
print("b4 grad at(" + str(ti) + "):" + str(grad_network_b4_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, ti)))

## traning network 

for i in range(60): # tranings rounds
    l = -0.02 # lering rate
    a1_bash = 0
    a2_bash = 0
    a3_bash = 0
    a4_bash = 0
    b1_bash = 0
    b2_bash = 0
    b3_bash = 0
    b4_bash = 0
    for j in range(10): #tranings bash size 
        a1_bash = a1_bash + l*grad_network_a1_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, i+j)
        a2_bash = a2_bash + l*grad_network_a2_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, i+j)
        a3_bash = a3_bash + l*grad_network_a3_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, i+j)
        a4_bash = a4_bash + l*grad_network_a4_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, i+j)
        b1_bash = b1_bash + l*grad_network_a1_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, i+j)
        b2_bash = b2_bash + l*grad_network_b2_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, i+j)
        b3_bash = b3_bash + l*grad_network_b3_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, i+j)
        b4_bash = b4_bash + l*grad_network_b4_error(a1 ,a2 ,a3, a4, b1, b2, b3, b4, i+j)
    a1 = a1 + a1_bash
    a2 = a2 + a2_bash
    a3 = a3 + a3_bash
    a4 = a4 + a4_bash
    b1 = b1 + b1_bash
    b2 = b2 + b2_bash
    b3 = b3 + b3_bash
    b4 = b4 + b4_bash

true_data = test_funktion(t)
estimated_data = network_1d(a1 ,a2 ,a3, a4, b1, b2, b3, b4 ,t)
error_data = error(estimated_data, true_data)

#sys.exit(1)
## Gradients
#grad_to_cartesian = jax.jacobian(to_cartesian)
#grad_to_cartesian_scalar = jax.grad(to_cartesian_scalar)

#g = grad_to_cartesian_scalar(1.0, 1.0)
#grad_data = grad_to_cartesian(polar_data)

### Plot data
#x_vals = jnp.array(cartesian_data[:, 0])
#y_vals = jnp.array(cartesian_data[:, 1])

plt.figure()
plt.plot(t, true_data, label="true")
plt.plot(t, estimated_data, label="estimated")
plt.plot(t, error_data, label="error")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Error plot")
plt.show()
