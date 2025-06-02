#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:49:54 2023

@author: vakilifard
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulation window parameters
h = 10  # height of cone
r = 5  # radius of base
xx0, yy0, zz0 = 0, 0, 35  # location of cone tip
volTotal = np.pi * r**2 * h / 3  # volume of cone

# Point process parameters
lambda_val = 1  # intensity (i.e., mean density) of the Poisson process

# Simulate Poisson point process
numbPoints = np.random.poisson(volTotal * lambda_val)  # Poisson number of points
zz = h * (np.random.rand(numbPoints, 1))**(1/3)  # z coordinates
theta = 2 * np.pi * (np.random.rand(numbPoints, 1))  # angular coordinates
rho = r * (zz / h) * np.sqrt(np.random.rand(numbPoints, 1))  # radial coordinates

# Convert from polar to Cartesian coordinates
xx = rho * np.cos(theta)
yy = rho * np.sin(theta)

# Shift tip of cone to (xx0, yy0, zz0)
xx += xx0
yy += yy0
zz += zz0

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_box_aspect([np.ptp(coord) for coord in [xx, yy, zz]])
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math

# Simulation window parameters
h = 10  # original height of the cone
r = 5   # radius of the base
xx0, yy0, zz0 = 0, 0, 35  # location of cone tip
h_trunc = h / 20  # height of the truncated cone
r_trunc = r * (h_trunc / h)  # radius at the truncation point

# Volume calculation for truncated cone
def truncated_cone_volume(r, h, r_trunc):
    v_full = (math.pi * r**2 * h) / 3
    v_small = (math.pi * r_trunc**2 * (h - h_trunc)) / 3
    v_truncated = v_full - v_small
    return v_truncated

# Calculate the volume of the truncated cone
volTotal = truncated_cone_volume(r, h, r_trunc)

# Point process parameters
lambda_val = 1  # intensity (i.e., mean density) of the Poisson process

# Simulate Poisson point process
numbPoints = np.random.poisson(volTotal * lambda_val)  # Poisson number of points
zz = h_trunc * (np.random.rand(numbPoints, 1))**(1/3)  # z coordinates limited to h/2
theta = 2 * np.pi * (np.random.rand(numbPoints, 1))  # angular coordinates
rho = (r * (zz / h)) * np.sqrt(np.random.rand(numbPoints, 1))  # radial coordinates adjusted

# Convert from polar to Cartesian coordinates
xx = rho * np.cos(theta)
yy = rho * np.sin(theta)

# Shift tip of cone to (xx0, yy0, zz0)
xx += xx0
yy += yy0
zz += zz0

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz, color='blue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_box_aspect([np.ptp(coord) for coord in [xx, yy, zz]])
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
import math

# Simulation window parameters
h = 600  # original height of the cone in km
r = 25   # radius of the base in km
xx0, yy0, zz0 = 0, 0, 600  # location of cone tip in km (apex of the cone)
h_trunc = 580  # height of the truncated cone
r_trunc = r * (h_trunc / h)  # radius at the truncation point

# Volume calculation for truncated cone
def truncated_cone_volume(r, h, r_trunc):
    v_full = (math.pi * r**2 * h) / 3
    v_small = (math.pi * r_trunc**2 * (h_trunc)) / 3
    v_truncated = v_full - v_small
    return v_truncated

# Calculate the volume of the truncated cone
volTotal = truncated_cone_volume(r, h, r_trunc)

# Point process parameters
lambda_val = 0.01  # intensity (i.e., mean density) of the Poisson process

# Simulate Poisson point process
numbPoints = np.random.poisson(volTotal * lambda_val)  # Poisson number of points

# z coordinates start at zz0 and move downward
zz = (zz0 - h_trunc) - (h-h_trunc) * (np.random.rand(numbPoints, 1))**(1/3)

# Angular coordinates for uniform distribution
theta = 2 * np.pi * (np.random.rand(numbPoints, 1))

# Radial coordinates, adjusted for downward-pointing cone
rho = (r_trunc / (h-h_trunc)) * ((zz0-h_trunc) - zz) * np.sqrt(np.random.rand(numbPoints, 1))

# Convert from polar to Cartesian coordinates
xx = rho * np.cos(theta)
yy = rho * np.sin(theta)

# Nodes locations
BS1_xx1 = 15; BS1_yy1 = 0; BS1_zz1 = 0
BS2_xx2 = 0; BS2_yy2 = -15; BS2_zz2 = 0
BS3_xx3 = -15; BS3_yy3 = 0; BS3_zz3 = 0
BS4_xx4 = 0; BS4_yy4 = 15; BS4_zz4 = 0
HAPS_xxh = 0; HAPS_yyh = 0; HAPS_zzh = 10
#odes = np.array([BS1_xx1,BS1_yy1,BS1_zz1], [BS2_xx2, BS2_yy2, BS2_zz2], [BS3_xx3, BS3_yy3, BS3_zz3], [BS4_xx4, BS4_yy4, BS4_zz4], [HAPS_xxh, HAPS_yyh, HAPS_zzh])
# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz, color='blue')
ax.scatter(BS1_xx1, BS1_yy1,BS1_zz1,  color = 'red')
ax.scatter(BS2_xx2, BS2_yy2, BS2_zz2,  color = 'red')
ax.scatter(BS3_xx3, BS3_yy3, BS3_zz3,  color = 'red')
ax.scatter(BS4_xx4, BS4_yy4, BS4_zz4,  color = 'red')
ax.scatter(HAPS_xxh, HAPS_yyh, HAPS_zzh,  color = 'red')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_box_aspect([np.ptp(coord) for coord in [xx, yy, zz]])
plt.show()



