import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# The desired reverberation time and dimensions of the room
rt60 = 0.4  # seconds
room_dim = [6.2, 7.5, 3.5]  # meters

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

# Create the room
room = pra.ShoeBox(
    room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order
)
