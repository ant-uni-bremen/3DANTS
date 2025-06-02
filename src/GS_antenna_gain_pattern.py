#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:51:11 2023

@author: vakilifard
"""

import numpy as np
import pandas as pd  # Import Pandas
import matplotlib.pyplot as plt
from scipy.special import j1, jv

# Define the antenna gain function
def antenna_gain_calc(theta):
    # Constants
    eta = 0.7
    N = 70
    theta_3dB = 5.12
    theta_3dB_rad = np.radians(theta_3dB)

    # Calculate u(theta)
    u_theta = 2.07123 * np.sin(np.radians(theta)) / np.sin(theta_3dB_rad)

    # Calculate G0
    G0 = (eta * N**2 * np.pi**2) / theta_3dB**2

    # Calculate G(theta)
    #G_theta = G0 * (j1(u_theta) / (5 * u_theta) + 30 * jv(3, u_theta) / u_theta**5)
    G_theta = G0 * (j1(u_theta) / (2 * u_theta) + 36 * jv(3, u_theta) / u_theta**3)

    # Convert gain to dB using 10*log10
    gain_dB = 10 * np.log10(G_theta)
    # Replace -inf values with a large negative number
    gain_dB = np.where(np.isnan(gain_dB), -50, gain_dB)

    return gain_dB

# Create an array of theta values from -90 to 90 degrees
theta_values = np.linspace(-50, 50, 1000)

# Calculate the gain for each theta value
gain_values = antenna_gain_calc(theta_values)

# Create a DataFrame to store the data
data = pd.DataFrame({'Theta (degrees)': theta_values, 'Gain (dB)': gain_values})

# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(theta_values, gain_values)
plt.title('Antenna Gain Pattern')
plt.xlabel('Theta (degrees)')
plt.ylabel('Gain (dB)')
plt.grid(True)

# Show the plot
plt.show()
