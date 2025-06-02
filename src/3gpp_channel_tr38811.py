#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:15:59 2023

@author: vakilifard
"""

import random
from re import L
import numpy as np
import math
from sympy import DiracDelta

class TR38811_Channel():
    
    # Step 1: Set environment, network layout, and antenna array parameters:

    def __init__(self, freq, elevation_angle, sat_position, gs_position, condition):
        self.freq = freq; self.elevation_angle = elevation_angle; self.sat_position = sat_position; self.gs_position = gs_position; self.condition= condition
        self.h_sat = sat_position[2]; self.h_UE = gs_position[2]
        self.distance_2D = np.linalg.norm(np.array([sat_position[0], sat_position[1], 0]) - np.array([gs_position[0], gs_position[1], 0]))
        ########### Array orientation angle 
        # It is considered that the BS and UT array orientations with respect to the global coordinate system is not changing and remains zero
        self.omega_BS_a = 0.0    #BS bearing angle
        self.omega_BS_b = 0.0    #BS downtilt angle
        self.omega_BS_c = 0.0    #BS slant angle
        self.omega_UT_a = 0.0    #UT bearing angle
        self.omega_UT_b = 0.0    #UT downtilt angle
        self.omega_UT_c = 0.0    #UT slant angle
        
        ################# Azimuth and Zenith angle of Los
        # It is considered that Give 3D locations of BS and UT, and determine LOS AOD (ϕLOS,AOD), LOS ZOD (θLOS,ZOD), LOS AO( ϕLOS,AOA), and LOS ZOA (θLOS,ZOA)
       # each BS and UT in the global coordinate system is considered zero.
        self.theta_LOS_ZOD = 0.0
        self.theta_LOS_ZOA = 0.0
        self.phi_LOS_AOD = 0.0
        self.phi_LOS_AOA = 0.0
        
        # Step 2: Assign propagation condition (LOS/NLOS) according to Table 7.4.2-1. It is calculated for each elevation angle. It is calculated in Rx_power_calc() class
        # Step 3: Calculate pathloss with formulas in section 6.6.2
        # Step4: 
            
        
        
    