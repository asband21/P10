import json
import sys
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

from scipy.io import wavfile

def to_cartesian_scalar(a, d):
    return jnp.cos(a)*d

def to_cartesian(polar):
    x = jnp.cos(polar[:,0])*polar[:,1]
    y = jnp.sin(polar[:,0])*polar[:,1]
    return jnp.array([x,y]).T

if len(sys.argv) != 4:
    print("Command must be in this form:\n script.py Wide_Long_angle_data.json Wide_Long_distance_data.json Wide_Long_sound.wav")
    sys.exit(1)

## Load data
angle = json.load(open(sys.argv[1]))
distance = json.load(open(sys.argv[2]))
sr, x = wavfile.read(sys.argv[3])

polar_data = jnp.array([angle["LiDAR_angle"], distance["LiDAR_distance"]]).T
audio_data = jnp.append(x[:,0], x[:,1])
print(audio_data.shape)

## Transform data
cartesian_data = to_cartesian(polar_data)

## Gradients
grad_to_cartesian = jax.jacobian(to_cartesian)
grad_to_cartesian_scalar = jax.grad(to_cartesian_scalar)

g = grad_to_cartesian_scalar(1.0, 1.0)
grad_data = grad_to_cartesian(polar_data)

sys.exit(1)

## Plot data
x_vals = jnp.array(cartesian_data[:, 0])
y_vals = jnp.array(cartesian_data[:, 1])

plt.figure()
plt.scatter(x_vals, y_vals)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("LiDAR XY Plot")
plt.axis("equal")
plt.show()
