#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:20:53 2024

@author: vakilifard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:09:28 2024

@author: vakilifard
"""

import numpy as np
import matplotlib.pyplot as plt
import math


# Volume calculation for truncated cone
def truncated_cone_volume(r, h, r_trunc):
    v_full = (math.pi * r**2 * h) / 3
    v_small = (math.pi * r_trunc**2 * (h_trunc)) / 3
    v_truncated = v_full - v_small
    return v_truncated


def HAPS_circular_trajectory(angular_velocity, radius, time_interval, center_position, step):
    # Simulate Circular trajectory
    x0, y0, z0 = center_position
    positions = []
    angle = step * angular_velocity * time_interval  # Convert to radians
    x = x0 + radius * np.cos(angle)
    y = y0 + radius * np.sin(angle)
    z = z0
    positions.append((x, y, z))
    
    return np.array(positions).reshape((3))


# Function to check if a point is inside a half-spheroid
def is_inside_half_spheroid(point, center, a, b, c):
    x, y, z = point - center
    if z < 0:  # Ensure it is above the z=0 plane
        return False
    return (x**2 / a**2 + y**2 / b**2 + z**2 / c**2) <= 1

# Function to check if a point is inside a cone
#def is_inside_cone(point, tip, radius, height):
#    x, y, z = point
#    if z < 0 or z > height:  # Ensure it's within the height of the cone
#        return False
#    cone_radius_at_z = (radius / height) * z
#    return (x**2 + y**2) <= cone_radius_at_z**2

def is_inside_cylinder(point, tip, radius, height):
    x, y, z = tip - point
    if z < 0 or z > height:  # Ensure it's within the height of the cone
        return False
    return (x**2 + y**2) <= radius**2


# General simulation parameters
Monte_Carlo_iteration = 2
rise_elevation_angle = 30 # in degree
set_elevation_angle = 35
elevation_angle_interval = set_elevation_angle - rise_elevation_angle
elevation_angle_array = np.linspace(rise_elevation_angle, set_elevation_angle, elevation_angle_interval)
channel_realization_per_TTI = 10000
all_data = [] # Initialization of storing data for the whole program

# Simulation window parameters
h = 600  # original height of the cone in km
r = 25   # radius of the base in km
xx0, yy0, zz0 = 0, 0, 600  # location of cone tip in km (apex of the cone)
h_trunc = 595  # height of the truncated cone
r_trunc = r * (h_trunc / h)  # radius at the truncation point

# Calculate the volume of the truncated cone
volTotal = truncated_cone_volume(r, h, r_trunc)

# Point process parameters
lambda_val = 0.1  # intensity (i.e., mean density) of the Poisson process

# Nodes locations
BS1 = np.array([15, 0, 0])
BS2 = np.array([0, -15, 0])
BS3 = np.array([-15, 0, 0])
BS4 = np.array([0, 15, 0])

# HAPS trajectory parameters
HAPS = np.array([0, 0, 10])
velocity_HAPS = 70 # km/hour
radius_HAPS = 2 #km
time_interval = 10 #second
angular_velocity = (velocity_HAPS / radius_HAPS) / (3600/time_interval)  # in radians/time_interval
number_step_haps = int(6.28319 / (angular_velocity * time_interval))  # Simulate one full circle
shiftak0_haps = 0

# Radii for spheroids and cone for HAPS
a = b = 10  # Semi-major axis for half-spheroids (BS1, BS2, BS3, BS4)
c = 5       # Semi-minor axis (height) for half-spheroids
HAPS_radius = 6  # Radius of the base of the HAPS cone
HAPS_height = 10 # Height of the HAPS cone

for sim in range(Monte_Carlo_iteration):
    Monte_Carlo_data = [] # Data for One Monte Cralo iteration
    for elev_loc, elev_ang in enumerate(elevation_angle_array):
        pass_data = {} # Data for one elevation angle realization
        pass_data['BS1'] = []
        pass_data['BS2'] = []
        pass_data['BS3'] = []
        pass_data['BS4'] = []
        pass_data['HAPS'] = []
        pass_data['satellite'] = []

        # Simulate Poisson point process inside the truncated cone as for satellite coverage space
        
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
        
        
        # Create arrays to store assigned points
        points_BS1 = []
        points_BS2 = []
        points_BS3 = []
        points_BS4 = []
        points_HAPS = []
        Satellite_points = []
        
        # HAPS center chnage due to its movement
        shiftak_haps = (elev_loc + shiftak0_haps) % number_step_haps
        step_haps = shiftak_haps + 1
        HAPS_center = HAPS_circular_trajectory(angular_velocity, radius_HAPS, time_interval, HAPS, step_haps)
        
        # Classify points based on proximity to BS nodes and HAPS
        for i in range(numbPoints):
            point = np.array([xx[i, 0], yy[i, 0], zz[i, 0]])
            if is_inside_half_spheroid(point, BS1, a, b, c):
                points_BS1.append(point)
            elif is_inside_half_spheroid(point, BS2, a, b, c):
                points_BS2.append(point)
            elif is_inside_half_spheroid(point, BS3, a, b, c):
                points_BS3.append(point)
            elif is_inside_half_spheroid(point, BS4, a, b, c):
                points_BS4.append(point)
            elif is_inside_cylinder(point, HAPS_center, HAPS_radius, HAPS_height):
                points_HAPS.append(point)
            else:
                Satellite_points.append(point)
        
        
        # Convert lists to arrays
        points_BS1 = np.array(points_BS1)
        points_BS2 = np.array(points_BS2)
        points_BS3 = np.array(points_BS3)
        points_BS4 = np.array(points_BS4)
        points_HAPS = np.array(points_HAPS)
        Satellite_points = np.array(Satellite_points)
        
        fig = plt.figure(figsize = (20,12))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xx, yy, zz,  marker="v" ,color='blue', alpha=0.1, label='All Poisson Points')
        ax.scatter(BS1[0], BS1[1], BS1[2],  marker="v", color='red', alpha=0.9, linewidths= 10, label='BS1')
        ax.scatter(BS2[0], BS2[1], BS2[2],  marker="v", color='red', alpha=0.9, linewidths= 10, label='BS2')
        ax.scatter(BS3[0], BS3[1], BS3[2],  marker="v", color='red', alpha=0.9, linewidths= 10, label='BS3')
        ax.scatter(BS4[0], BS4[1], BS4[2],  marker="v", color='red', alpha=0.9, linewidths= 10, label='BS4')
        ax.scatter(HAPS[0], HAPS[1], HAPS[2], marker="8", color='green', alpha=0.9, linewidths= 10, label='HAPS')
        
        # Plot assigned points in different colors
        if points_BS1.size > 0:
            ax.scatter(points_BS1[:, 0], points_BS1[:, 1], points_BS1[:, 2] ,color='purple', label='BS1 Points')
        if points_BS2.size > 0:
            ax.scatter(points_BS2[:, 0], points_BS2[:, 1], points_BS2[:, 2],  color='orange', label='BS2 Points')
        if points_BS3.size > 0:
            ax.scatter(points_BS3[:, 0], points_BS3[:, 1], points_BS3[:, 2], color='yellow', label='BS3 Points')
        if points_BS4.size > 0:
            ax.scatter(points_BS4[:, 0], points_BS4[:, 1], points_BS4[:, 2],  color='cyan', label='BS4 Points')
        if points_HAPS.size > 0:
            ax.scatter(points_HAPS[:, 0], points_HAPS[:, 1], points_HAPS[:, 2] ,color='magenta', label='HAPS Points')
        if Satellite_points.size > 0:
            ax.scatter(Satellite_points[:, 0], Satellite_points[:, 1], Satellite_points[:, 2], color='black', label='Satellite Points')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend(loc="lower left")
        ax.set_box_aspect([np.ptp(coord) for coord in [xx, yy, zz]])
        #plt.savefig("figure.pgf")
        plt.show()