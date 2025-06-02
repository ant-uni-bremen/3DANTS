#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:25:08 2024

@author: vakilifard
"""


import numpy as np


class HAPS_trajectory:
    
    
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
        sin_angle = (z0 - GS_position[2])/np.linalg.norm(v)
        elev_angle_rad = np.arcsin(np.clip(sin_angle, -1.0, 1.0))
        elev_angle_deg = np.rad2deg(elev_angle_rad)
        if elev_angle_deg > 90:
            elev_angle_deg = 180 - elev_angle_deg
        positions.append((x, y, z, elev_angle_deg))
        
        return np.array(positions)