import sys
import jax
import time
import matplotlib.pyplot as plt
import jax.numpy as jnp

def test_funktion(x):
    return 0.1*x*x + x*1.8 - 0.5

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
    return x

def net_str_make(n , relu_bool):
    rl = relu_bool
    s = " neuron( a0, b0, x)"
    for i in range(1,n):
        if rl:
            s = f"jax.nn.relu({s})"
        s = f" neuron( a{i}, b{i}, {s})"
    s = " x :" + s
    for i in range(n):
        s = ' a' + str(i) + ", b" + str(i) + "," + s
    s = "lambda " + s
    return s

def arret_net_fun_maker(n):
    s = ""
    for i in range(n):
        s = ' a[' + str(i) + "], b[" + str(i) + "]," + s

    s = "lambda a, b, x, f : f(" + s + " x)"
    f = eval(s)
    return f

def arret_grad_net_fun_maker(n):
    s = ""
    for i in range(n):
        s = ' a[' + str(i) + "], b[" + str(i) + "]," + s

    s = "lambda a, b, x, f : f(" + s + " x)"
    f = eval(s)
    return f

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

def gradien_funksen_list(n , net):
    nabla = []
    for i in range(n):
        nabla.append(jax.grad(net, i*2))
        nabla.append(jax.grad(net, i*2 + 1))
    return nabla

def get_gredent(a, b, x, n, f, gradind_funksen_list, arrey_funksen):
    print(f"a:{a}, b:{b}, x:{x}, n:{n}, gradind_funksen_list:{gradien_funksen_list}")
    gr = jnp.zeros(n*2)
    for i in range(n):
        #gr = gr.at[i*2].set(gradind_funksen_list[i*2](a, b, x, f))
        #gr = gr.at[i*2 + 1].set(gradind_funksen_list[i*2 + 1](a, b, x, f))
        
        #print(gradind_funksen_list[i*2](a, b, x, f))
        #print(gradind_funksen_list[i*2 + 1](a, b, x, f))
        
        print(arrey_funksen(a, b, x, gradind_funksen_list[i*2]) )
        print(arrey_funksen(a, b, x, gradind_funksen_list[i*2 + 1]))

        #gr = gr.at[i*2].set(arrey_funksen(a, b, x, gradind_funksen_list[i*2]) )
        #gr = gr.at[i*2 + 1].set(arrey_funksen(a, b, x, gradind_funksen_list[i*2 + 1]) )
        #gr.at[i*2].set[i*2](arrey_funksen(a, b, x, gradind_funksen_list[i*2]))
        #gr.at[i*2 +1].set[i*2 +1](arrey_funksen(a, b, x, gradind_funksen_list[i*2 + 1]))
    return gr

n = 30 # nummer of 1d neuron lager

print("--------------")

bool_relu = True
print(net_str_make(n, bool_relu))

f = eval(net_str_make(n, bool_relu))

#print(f(3.0, 4.0, 3.0, 4.0, 3.0, -4.0, 1.0))
#gd_f = jax.grad(f, 1)
#print(gd_f(3.0, 4.0, 3.0, 4.0, 3.0, -4.0, 1.0))

key = jax.random.key(int(time.time()))
#key = jax.random.key(10)
key, k1 = jax.random.split(key)
a = jax.random.uniform(k1, shape=(n), minval=-1.0, maxval=1.0)

key, k2 = jax.random.split(key)
b = jax.random.uniform(k2, shape=(n), minval=-1.0, maxval=1.0)

aa_net_fun = arret_net_fun_maker(n)
print(aa_net_fun)
print(aa_net_fun(a, b, 2.2 , f))


gra_fun_list = gradien_funksen_list(n , aa_net_fun)
gra_fun_list = gradien_funksen_list(n , f)

print(gra_fun_list[1])
print("print v gradiend")
#print(gra_fun_list[1](a, b, 2.2, f))
print("----- A --------")



#sys.exit(1)


ggg = get_gredent(a, b, 2.2, n, f, gra_fun_list, aa_net_fun)
print(ggg)




sys.exit(1)
## maked the gradient funksen


grad_network_a_error = [] 
grad_network_b_error = []

for i in range(n):
    grad_network_a_error.append(jax.grad(net, n))
    grad_network_a_error.append(jax.grad(net, n+1))

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



