import json
import sys
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

from scipy.io import wavfile



def til_kartikes_en(a, d):
    return jnp.cos(a)*d

def til_kartikes(polært):
    x = jnp.cos(polært[:,0])*polært[:,1]
    y = jnp.sin(polært[:,0])*polært[:,1]
    return jnp.array([x,y]).T

if len(sys.argv) != 3:
    print("comandoen skal være på denne form:\n shtipt.py Wide_Long_angle_data.json Wide_Long_distance_data.json")
    sys.exit(1)

## hender data 
vinkel = json.load(open(sys.argv[1]))
lengte = json.load(open(sys.argv[2]))

pol_data = jnp.array([vinkel["LiDAR_angle"], lengte["LiDAR_distance"]]).T

## trens former data:
kartesisk_data = til_kartikes(pol_data)
# print(jax.make_jaxpr(til_kartikes)(pol_data)) til at kigge på JIT'en

grad_til_kartikes = jax.jacobian(til_kartikes)
grad_til_kartikes_en = jax.grad(til_kartikes_en)

print(grad_til_kartikes)
print(grad_til_kartikes_en)


g = grad_til_kartikes_en(1.0,1.0)
grad_data = grad_til_kartikes(pol_data)

print(g)
print(grad_data)

#grad_data = grad_til_kartikes(pol_data[:,0])
#grad_data = grad_til_kartikes(pol_data)
#grad_data = grad_til_kartikes(pol_data[:,0])

## plotter det 
x = jnp.array(kartesisk_data[:,0])
y = jnp.array(kartesisk_data[:,1])

plt.figure()
plt.scatter(x, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("LiDAR XY Plot")
plt.axis("equal")
plt.show()
