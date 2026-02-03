import json
import sys
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

def til_kartikes(polært):
    return jnp.array([jnp.cos(polært[0])*polært[1], jnp.sin(polært[0])*polært[1]])

if len(sys.argv) != 3:
    print("comandoen skal være på denne form:\n shtipt.py Wide_Long_angle_data.json Wide_Long_distance_data.json")
    sys.exit(1)

print(sys.argv[1])
vinkel = json.load(open(sys.argv[1]))

print(sys.argv[2])
lengte = json.load(open(sys.argv[2]))

pol_data = jnp.array([vinkel["LiDAR_angle"], lengte["LiDAR_distance"]]).T

kartesisk_data = jax.vmap(til_kartikes, in_axes=0)(pol_data)

x = jnp.array(kartesisk_data[:,0])
y = jnp.array(kartesisk_data[:,1])

plt.figure()
plt.scatter(x, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("LiDAR XY Plot")
plt.axis("equal")
plt.show()
