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
from numpy import linalg as LA
import datetime
from sgp4.api import Satrec, WGS72
from skyfield.api import load, wgs84, Topos
import skyfield.api as sf
from GEO import LEO_GEO
from Satellite_comm_param import Satellite_communication_parameter
from Air_objects_class import Air
from Uav_trajec_class import Uav_trajectory
from HAPS_trajec_class import HAPS_trajectory
from Terresterial_Object import terresterial_network
import matplotlib.pyplot as plt
import pandas as pd
import copy
import os


if __name__ == '__main__':
    
    np.random.seed(42)
    #%%
    """ ################################################# Parameters intialization for Satellite ################################################################ """
    
    # constellation first
    h_LEO_1 = 600e3
    inclination_1 = 60
    numSat_1 = 60
    numPlanes_1 = 10
    # constellation second
    h_LEO_2 = 600e3
    inclination_2 = 60
    numSat_2 = 60
    numPlanes_2 = 20
    ### With this setting each 10 minutes we have three satellites ###
    phasing = 1
    r_E = 6371e3
    gm = 3.986004418e14
    h_GEO = 20000e3
    
    ###################                                          Satellite Transmission parameters                                ##################
    #################### This Parameters are based on 3gpp TR 38.821 chapter 6 --> Tables 6.1.1.1-1 to 6.1.1.1-6 ############################
    f = 2.0e9;
    satellite_EIRP_density, satellite_Tx_max_Gain, satellite_3dB_beamwidth, satellite_beam_diameter,  max_Bandwidth_per_beam = Satellite_communication_parameter().parameters(f, 'S', 'DL')
    #satellite_EIRP = satellite_Tx_max_Gain + satellite_EIRP_density + 10*np.log10(max_Bandwidth_per_beam)
    satellite_EIRP = satellite_Tx_max_Gain + satellite_EIRP_density    
    ##### creation of LEO Walker constellation ####
    LG = LEO_GEO(r_E, gm, h_GEO)
    LEOs_1 = LG.walkerConstellation(h_LEO_1, inclination_1, numSat_1, numPlanes_1, phasing, name = "Sat")
    orbital_speed = LG.motion(r_E+h_LEO_1, 'meter/sec')/1000
    LEOs_2 = LG.walkerConstellation(h_LEO_2, inclination_2, numSat_2, numPlanes_2, phasing, name = "Sat")
    orbital_speed_2 = LG.motion(r_E+h_LEO_2, 'meter/sec')/1000


    ##################### Seperate the satellites in each plane ##########################
    LEO_sats_in_plane_1 = []
    for i in range(numPlanes_1):
        a = int(numSat_1/numPlanes_1)
        LEO_sats_in_plane_1.append(LEOs_1[i*a: a + i * a])


    ############################start and end time of satellites simulation#################################

    ts = sf.load.timescale()
    time1 = ts.utc(2022,9,22,00,00,00)      # Start point, UTC time
    time2 = ts.utc(2022,9,22,18,00,00)      # End point, UTC time
    seconds_difference_time = LG.difference_time_in_seconds(time1, time2) # How many seconds between start and end

    ####################################*********** Create the constellation and calculate the repsected parameters **************##########################################

    groundstation = wgs84.latlon(53.110987, 8.851239) #GS-BR
    GS_distance_from_Earth_center = LA.norm(groundstation.itrs_xyz.km)
    DF_1 = LG.simulateConstellation(LEOs_1, groundstation, 20, time1, time2, ts = None, safetyMargin = 0)
    DF2_const1 = DF_1.reset_index()
    DF_2 = LG.simulateConstellation(LEOs_2, groundstation, 20, time1, time2, ts = None, safetyMargin = 0)
    DF2_const2 = DF_2.reset_index()
    # Calculate total visibility
    total_visibility1 = DF2_const1.groupby('Satellite')['Visibility'].sum()
    total_visibility1_minutes = total_visibility1.dt.total_seconds() / 60
    satellite_order1 = DF2_const1.groupby('Satellite')['Rise'].min().sort_values().index
    total_visibility1_minutes = total_visibility1_minutes.loc[satellite_order1]
    
    total_visibility2 = DF2_const2.groupby('Satellite')['Visibility'].sum()
    total_visibility2_minutes = total_visibility2.dt.total_seconds() / 60
    satellite_order2 = DF2_const2.groupby('Satellite')['Rise'].min().sort_values().index
    total_visibility2_minutes = total_visibility2_minutes.loc[satellite_order2]
    
    plt.figure(figsize=(18, 12), dpi = 1024)
    plt.bar(total_visibility1_minutes.index, total_visibility1_minutes.values, alpha=0.5, label='Constellation 1')
    plt.bar(total_visibility2_minutes.index, total_visibility2_minutes.values, alpha=0.5, label='Constellation 2')
    plt.xlabel('Satellite Name')
    plt.ylabel('Total Visibility Duration (minutes)')
    plt.title('Visibility Duration Comparison')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
