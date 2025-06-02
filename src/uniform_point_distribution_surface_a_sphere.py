#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:23:13 2024

@author: vakilifard
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up random number generator with system time as seed
np.random.seed(None)

# Generate N random numbers
N = 1
# defien center
x_center, y_center, z_center = np.array([0,0,0])
# Lists to store Cartesian coordinates
x_values, y_values, z_values = [], [], []

# Sphere radius
radius = 5.0

# Generate points
for i in range(N):
    # Generate random angles
    theta = 2 * math.pi * np.random.uniform()
    phi = math.acos(1 - 2 * np.random.uniform())

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * math.sin(phi) * math.cos(theta) + x_center
    y = radius * math.sin(phi) * math.sin(theta) + y_center
    z = radius * math.cos(phi) + z_center

    # Append coordinates to the lists
    x_values.append(x)
    y_values.append(y)
    z_values.append(z)

# Plot the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_center, y_center, z_center, s=5, c='red', marker='o')
ax.scatter(x_values, y_values, z_values, s=5, c='blue', marker='o')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
