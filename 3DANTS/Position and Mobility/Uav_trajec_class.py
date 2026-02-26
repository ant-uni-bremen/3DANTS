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
import math

class Uav_trajectory:
    
    
    def __init__(self, velocity, radius, time_interval):
        self.velocity = velocity
        self.radius = radius
        self.angular_velocity = (self.velocity / self.radius) / 3600  # in radians/second
        self.time_interval = time_interval
        self.num_steps = int(6.28319 / (self.angular_velocity * self.time_interval))  # Simulate one full circle

    def get_values(self):
        return self.angular_velocity, self.num_steps
    
    
    def simulate_circular_trajectory(self, center_position ,h0, step, GS_position):
        
        self.step = step
        # Simulate Circular trajectory
        x0, y0, z0 = center_position
        positions = []
        angle = self.step * self.angular_velocity * self.time_interval  # Convert to radians
        x = x0 + self.radius * np.cos(angle)
        y = y0 + self.radius * np.sin(angle)
        z = z0
        v = GS_position - np.array([x, y, z])
        sin_angle = (h0)/np.linalg.norm(v)
        elev_angle_rad = np.arcsin(np.clip(sin_angle, -1.0, 1.0))
        elev_angle_deg = np.rad2deg(elev_angle_rad)
        positions.append((x, y, z, elev_angle_deg))
        
        return np.array(positions)

    def UAV_trajectory_CIM(self, center_position, speed, time_step, reference_point, GS_position):
        # Generate N random numbers
        N = 1
        # defien center
        x_center, y_center, z_center = center_position
        # Lists to store Cartesian coordinates
        positions = []

        # Sphere radius
        radius = (speed/3.6) * time_step #convert it to meter/second
        if z_center < GS_position[2] + 0.1:
            z_center = GS_position[2] + 0.11
        # Generate points
        for i in range(N):
            # Generate random angles
            theta = 2 * math.pi * np.random.uniform()
            phi = math.acos(1 - 2 * np.random.uniform())

            # Convert spherical coordinates to Cartesian coordinates
            x = radius * math.sin(phi) * math.cos(theta) + x_center
            y = radius * math.sin(phi) * math.sin(theta) + y_center
            z = radius * math.cos(phi) + z_center
            v = reference_point - np.array([x, y, z])
            sin_angle = (reference_point[2]-z)/np.linalg.norm(v)
            elev_angle_rad = np.arcsin(np.clip(sin_angle, -1.0, 1.0))
            elev_angle_deg = np.rad2deg(elev_angle_rad)
            positions.append((x, y, z, elev_angle_deg))
            positions = np.array(positions)
            return positions 
