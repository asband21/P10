import sys
import jax
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    Cardioid,
    DirectionVector,
)
from scipy.io import wavfile


key = jax.random.key(int(time.time()))
key, k1 = jax.random.split(key)

number_of_mic_rum = 10
room_dimesen = jax.random.uniform(k1, shape=(number_of_mic_rum, 3), minval=5.0, maxval=10.0)
print(room_dimesen[0])



# The desired reverberation time and dimensions of the room
rt60_tgt = 0.3  # seconds
room_dim = room_dimesen[0].tolist()


# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
fs, audio = wavfile.read("damer.wav")

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

# Create the room
room = pra.ShoeBox(
    room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
)

# Source points along +x direction, horizontally
src_orientation = DirectionVector(azimuth=0, colatitude=90, degrees=True)
src_directivity = Cardioid(orientation=src_orientation)

# place the source in the room
room.add_source([2.5, 3.73, 1.76], signal=audio, delay=0.5)

# define the locations of the microphones

mic_locs = np.c_[
    [6.3, 4.87, 1.2], [6.3, 4.93, 1.2],  # mic 1  # mic 2
]
#mic_locs = np.c_[[6.3, 4.87, 1.2] , ]

# finally place the array in the room
room.add_microphone_array(mic_locs)

# Run the simulation (this will also build the RIR automatically)
room.simulate()

room.mic_array.to_wav(
    "damer_reverb.wav",
    norm=True,
    bitdepth=np.int16,
)

#sys.exit()
# measure the reverberation time
rt60 = room.measure_rt60()
print("The desired RT60 was {}".format(rt60_tgt))
print("The measured RT60 is {}".format(rt60[1, 0]))

# Create a plot
plt.figure()

# plot one of the RIR. both can also be plotted using room.plot_rir()
rir_1_0 = room.rir[1][0]
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
plt.title("The RIR from source 0 to mic 1")
plt.xlabel("Time [s]")

# plot signal at microphone 1
plt.subplot(2, 1, 2)
plt.plot(room.mic_array.signals[1, :])
plt.title("Microphone 1 signal")
plt.xlabel("Time [s]")

plt.tight_layout()
plt.show()
