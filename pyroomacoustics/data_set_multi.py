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
#rum høj 2.3 m til 4 m
#rum mick høj .03
# min dim 3 m  
room_dimesen = jax.random.uniform(k1, shape=(number_of_mic_rum, 3), minval=3, maxval=10.0)

key, k3 = jax.random.split(key)
mic_positions = jax.random.uniform(k3, shape=(number_of_mic_rum, 3), minval=jnp.zeros((number_of_mic_rum, 3)), maxval=room_dimesen,)

# The desired reverberation time and dimensions of the room
fs, audio = wavfile.read("Noctule_kort.wav")
rt60_tgt = 0.8  # seconds


for i in range(number_of_mic_rum):
    room_dim = room_dimesen[i].tolist()
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    # Create the room
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)

    # Source points along +x direction, horizontally
    src_orientation = DirectionVector(azimuth=0, colatitude=90, degrees=True)
    src_directivity = Cardioid(orientation=src_orientation)
    room.add_source(mic_positions[i], signal=audio, delay=0.5)

    # define the locations of the microphones

    dl = mic_positions[i].tolist() 
    dl_1 = dl.copy() 
    dl_2 = dl.copy()
    dl_1[1] = dl[1] + 0.05 
    dl_2[1] = dl[1] - 0.05 
    print(dl_1)
    print(dl_2)

    mic_locs = np.c_[
        dl_1, dl_2,  # mic 1  # mic 2
    ]
    room.add_microphone_array(mic_locs)

    room.simulate()

    room.mic_array.to_wav(
        f"echo/Noctule_50hz_tail_echo_{i}.wav",
        norm=True,
        bitdepth=np.int16,
    )

    # measure the reverberation time
    #rt60 = room.measure_rt60()
    #print("The desired RT60 was {}".format(rt60_tgt))
    #print("The measured RT60 is {}".format(rt60[1, 0]))

sys.exit()
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
