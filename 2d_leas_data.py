import json
import sys
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp


def til_kartikes(a, d):
    x = jnp.cos(a)*d
    y = jnp.sin(a)*d
    return x, y

if len(sys.argv) != 3:
    print("comandoen skal være på denne form:\n shtipt.py Wide_Long_angle_data.json Wide_Long_distance_data.json")
    sys.exit(1)

## hender data 
vinkel = json.load(open(sys.argv[1]))
lengte = json.load(open(sys.argv[2]))

pol_data = jnp.array([vinkel["LiDAR_angle"], lengte["LiDAR_distance"]]).T

a = pol_data[:, 0];
d = pol_data[:, 1];

## trens former data:
x, y = til_kartikes(a ,d)

## dif
grad_til_kartikes = jax.grad(til_kartikes)
gred_x  = grad_til_kartikes(a , d)

## plotter det 
#x = jnp.array(kartesisk_data[:,0])
#y = jnp.array(kartesisk_data[:,1])

plt.figure()
plt.scatter(x, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("LiDAR XY Plot")
plt.axis("equal")
plt.show()
