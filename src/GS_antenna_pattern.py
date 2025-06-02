#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:37:36 2023

@author: vakilifard
"""
#"""
import numpy as np
from scipy.special import j1, jv
import matplotlib.pyplot as plt

def simulate_gain(theta):
    # Constants
    eta = 0.7
    N = 70
    theta_3dB = 5.12
    theta_3dB_rad = np.radians(theta_3dB)

    # Calculate u(theta)
    u_theta = 2.07123 * np.sin(np.radians(theta)) / np.sin(theta_3dB_rad)

    # Calculate G0
    G0 = (eta * N**2 * np.pi**2) / theta_3dB**2
    #G0 = 10**(35/10)

    # Calculate G(theta)
    G_theta = G0 * (j1(u_theta) / (2 * u_theta) + 36 * jv(3, u_theta) / u_theta**3)

    return G_theta
#
f = 20e9
# Range of off-axis angles in degrees
theta_range = np.linspace(-90, 90, num=180)

# Calculate the gain for each angle
gain_values = simulate_gain(theta_range)

# Convert gain to dB using 10*log10
gain_dB = 10 * np.log10(gain_values)
# Replace -inf values with a large negative number
gain_dB = np.where(np.isnan(gain_dB), -60, gain_dB)

# Plot the gain vs off-axis angle in dB scale
plt.plot(theta_range, gain_dB)
plt.xlabel("Off-axis Angle (degrees)")
plt.ylabel("Gain (dBi)")
plt.title("Antenna Gain vs Off-axis Angle (dBi)")
plt.grid(True)
plt.show()
"""
import numpy as np
from scipy.special import j1, jv
import matplotlib.pyplot as plt

def simulate_gain(theta):
    # Constants
    eta = 0.7
    N = 65
    theta_3dB = 5.12
    theta_3dB_rad = np.radians(theta_3dB)

    # Calculate u(theta)
    u_theta = 2.07123 * np.sin(np.radians(theta)) / np.sin(theta_3dB_rad)

    # Calculate G0
    G0 = (eta * N**2 * np.pi**2) / theta_3dB_rad**2

    # Calculate G(theta)
    G_theta = G0 * (j1(u_theta) / (2 * u_theta) + 36 * jv(3, u_theta) / u_theta**3)

    return G_theta

# Range of off-axis angles in degrees
theta_range = np.linspace(-90, 90, num=361)

# Calculate the gain for each angle
gain_values = simulate_gain(theta_range)

# Convert gain to dB using 10*log10
gain_dB = 10 * np.log10(gain_values)

# Replace -inf values with a large negative number
gain_dB = np.where(np.isnan(gain_dB), -60, gain_dB)

# Plot the gain vs off-axis angle in dB scale
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.plot(np.radians(theta_range), gain_dB)
ax.set_rticks([-100, -80, -60, -40, -20, 0])
ax.set_rlabel_position(135)
ax.grid(True)

plt.title("Antenna Gain vs Off-axis Angle (dB)")
plt.show()
"""