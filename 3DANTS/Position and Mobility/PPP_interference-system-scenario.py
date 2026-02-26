#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file was created by the Department of Communications Engineering,
University of Bremen, Germany.
https://github.com/ant-uni-bremen
Copyright (c) 2026 Department of Communications Engineering, University of Bremen
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def is_inside_cylinder(point, tip, radius, height):
    x, y, z = tip - point
    if z < 0 or z > height:  # Ensure it's within the height of the cone
        return False
    return (x**2 + y**2) <= radius**2


# Function to create a cylinder
def create_cylinder(center_x, center_y, bottom_z, radius, height, resolution=20):
    z = np.linspace(bottom_z, bottom_z + height, 2)
    theta = np.linspace(0, 2*np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    
    # Create the vertices for the cylinder
    verts = []
    for i in range(resolution-1):
        verts.append([
            (x_grid[0, i], y_grid[0, i], z_grid[0, i]),
            (x_grid[0, i+1], y_grid[0, i+1], z_grid[0, i+1]),
            (x_grid[1, i+1], y_grid[1, i+1], z_grid[1, i+1]),
            (x_grid[1, i], y_grid[1, i], z_grid[1, i])
        ])
    
    # Create top and bottom circles
    circle_top = []
    circle_bottom = []
    for i in range(resolution-1):
        circle_top.append((x_grid[1, i], y_grid[1, i], z_grid[1, i]))
        circle_bottom.append((x_grid[0, i], y_grid[0, i], z_grid[0, i]))
    
    verts.append(circle_top)
    verts.append(circle_bottom)
    
    return verts


# General simulation parameters
Monte_Carlo_iteration = 10
rise_elevation_angle = 59 # in degree
set_elevation_angle = 60
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
lambda_val = 0.01  # intensity (i.e., mean density) of the Poisson process

# Nodes locations - Now with just 2 base stations
BS1 = np.array([15, 0, 0])
BS2 = np.array([-15, 0, 0])

# HAPS trajectory parameters
HAPS = np.array([0, 0, 10])
velocity_HAPS = 70 # km/hour
radius_HAPS = 2 #km
time_interval = 10 #second
angular_velocity = (velocity_HAPS / radius_HAPS) / (3600/time_interval)  # in radians/time_interval
number_step_haps = int(6.28319 / (angular_velocity * time_interval))  # Simulate one full circle
shiftak0_haps = 0

# Radii for spheroids and cylinder for HAPS
a = b = 10  # Semi-major axis for half-spheroids (BS1, BS2)
c = 5       # Semi-minor axis (height) for half-spheroids
HAPS_radius = 6  # Radius of the base of the HAPS cylinder
HAPS_height = 10 # Height of the HAPS cylinder

#for sim in range(Monte_Carlo_iteration):
for sim in range(0,1):
    Monte_Carlo_data = [] # Data for One Monte Carlo iteration
    for elev_loc, elev_ang in enumerate(elevation_angle_array):
        pass_data = {} # Data for one elevation angle realization
        pass_data['BS1'] = []
        pass_data['BS2'] = []
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
        points_HAPS = []
        Satellite_points = []
        
        # HAPS center change due to its movement
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
            elif is_inside_cylinder(point, HAPS_center, HAPS_radius, HAPS_height):
                points_HAPS.append(point)
            else:
                Satellite_points.append(point)
        
        
        # Convert lists to arrays
        points_BS1 = np.array(points_BS1) if points_BS1 else np.empty((0, 3))
        points_BS2 = np.array(points_BS2) if points_BS2 else np.empty((0, 3))
        points_HAPS = np.array(points_HAPS) if points_HAPS else np.empty((0, 3))
        Satellite_points = np.array(Satellite_points) if Satellite_points else np.empty((0, 3))
        
        # Create 3D figure with adjusted viewing angle to emphasize HAPS height
        fig = plt.figure(figsize=(12, 11), dpi = 1000)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set viewing angle to better show the HAPS height
        ax.view_init(elev=15, azim=45)  # Adjust these values as needed
        
        # Plot all points and base stations
        ax.scatter(xx, yy, zz, marker="v", color='blue', alpha=0.1)
        ax.scatter(BS1[0], BS1[1], BS1[2], marker="s", color='red', alpha=0.9, s=100)
        ax.scatter(BS2[0], BS2[1], BS2[2], marker="s", color='red', alpha=0.9, s=100, label='BS')
        ax.scatter(HAPS_center[0], HAPS_center[1], HAPS_center[2], marker="8", color='green', alpha=0.9, s=100, label='HAPS')
        
        # Plot assigned points in different colors
        if points_BS1.size > 0:
            ax.scatter(points_BS1[:, 0], points_BS1[:, 1], points_BS1[:, 2], color='red')
            
            # Create and add transparent cylinder for BS1
            cylinder_height = c  # Height based on the half-spheroid height
            cylinder_verts = create_cylinder(BS1[0], BS1[1], 0, a, cylinder_height)
            cyl_collection = Poly3DCollection(cylinder_verts, alpha=0.05, color='red', linewidths=1, edgecolor='red')
            ax.add_collection3d(cyl_collection)
            
            # Add radius arrow and text
            arrow_x = np.array([BS1[0], BS1[0] + a])
            arrow_y = np.array([BS1[1], BS1[1]])
            arrow_z = np.array([0, 0])
            ax.plot(arrow_x, arrow_y, arrow_z, color='red', linestyle='--', linewidth=2)
            ax.text((BS1[0] + a/2), BS1[1]+4, 0, f"r={a}km", color='red', fontsize=20)
            ax.text((BS1[0] + a/2), BS1[1]+4.5, 5, f"BS1", color='red', fontsize=20)
            
        if points_BS2.size > 0:
            ax.scatter(points_BS2[:, 0], points_BS2[:, 1], points_BS2[:, 2], color='red', label='UEs of BSs')
            
            # Create and add transparent cylinder for BS2
            cylinder_height = c  # Height based on the half-spheroid height
            cylinder_verts = create_cylinder(BS2[0], BS2[1], 0, a, cylinder_height)
            cyl_collection = Poly3DCollection(cylinder_verts, alpha=0.05, color='red', linewidths=1, edgecolor='red')
            ax.add_collection3d(cyl_collection)
            
            # Add radius arrow and text
            arrow_x = np.array([BS2[0], BS2[0] + a])
            arrow_y = np.array([BS2[1], BS2[1]])
            arrow_z = np.array([0, 0])
            ax.plot(arrow_x, arrow_y, arrow_z, color='red', linestyle='--', linewidth=2)
            ax.text((BS2[0] + a/2), BS2[1]+4.5, 0, f"r={a}km", color='red', fontsize=20)
            ax.text((BS2[0] + a/2), BS2[1]+4, 5, f"BS2", color='red', fontsize=20)

        if points_HAPS.size > 0:
            ax.scatter(points_HAPS[:, 0], points_HAPS[:, 1], points_HAPS[:, 2], color='green', label='UEs of HAPS')
            
            # Create and add transparent cylinder for HAPS
            cylinder_verts = create_cylinder(HAPS_center[0], HAPS_center[1], 0, HAPS_radius, HAPS_height)
            cyl_collection = Poly3DCollection(cylinder_verts, alpha=0.05, color='green', linewidths=1, edgecolor='green')
            ax.add_collection3d(cyl_collection)
            
            # Add radius arrow and text
            arrow_x = np.array([HAPS_center[0], HAPS_center[0] + HAPS_radius])
            arrow_y = np.array([HAPS_center[1], HAPS_center[1]])
            arrow_z = np.array([HAPS_center[2], HAPS_center[2]])
            ax.plot(arrow_x, arrow_y, arrow_z, color='green', linestyle='--', linewidth=2)
            ax.text((HAPS_center[0] + HAPS_radius/2), HAPS_center[1]+4, HAPS_center[2], f"r={HAPS_radius}km", color='green', fontsize=20)
            ax.text((HAPS_center[0] + HAPS_radius/2), HAPS_center[1]+4, HAPS_center[2]+1, f"HAPS", color='green', fontsize=20)

        if Satellite_points.size > 0:
            ax.scatter(Satellite_points[:, 0], Satellite_points[:, 1], Satellite_points[:, 2], color='blue', label='UEs of Satellite')
        
        # Add labels and legend
        ax.set_xlabel('x (km)', fontsize=20)
        ax.set_ylabel('y (km)', fontsize=20)
        ax.set_zlabel('Distance from ground (km)', fontsize=20)
        #ax.legend(loc="lower left")
        #ax.legend(loc="lower left", fontsize=12, markerscale=1.5, borderpad=1.5, frameon=True)
        # Set title
        #plt.title('Network Coverage Simulation - Elevation Angle: {:.1f}°'.format(elev_ang))
        plt.tight_layout()
        plt.savefig("/home/vakilifard/Documents/codes_result/saved_data_from_simulations/3D-scenario-PPP/network_simulation.jpg", format="jpg", dpi=1000, bbox_inches="tight")
        plt.show()
