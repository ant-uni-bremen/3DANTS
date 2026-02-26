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
    
    h_LEO = 600e3
    inclination = 60
    numSat = 120
    numPlanes = 10
    ### With this setting each 10 minutes we have three satellites ###
    phasing = 1
    r_E = 6371e3
    gm = 3.986004418e14
    h_GEO = 20000e3
    
    ###################                                          Satellite Transmission parameters                                ##################
    #################### This Parameters are based on 3gpp TR 38.821 chapter 6 --> Tables 6.1.1.1-1 to 6.1.1.1-6 ############################
    f = 2.5e9;
    satellite_EIRP_density, satellite_Tx_max_Gain, satellite_3dB_beamwidth, satellite_beam_diameter,  max_Bandwidth_per_beam = Satellite_communication_parameter().parameters(f, 'S', 'DL')
    #satellite_EIRP = satellite_Tx_max_Gain + satellite_EIRP_density + 10*np.log10(max_Bandwidth_per_beam)
    satellite_EIRP = satellite_Tx_max_Gain + satellite_EIRP_density    
    ##### creation of LEO Walker constellation ####
    LG = LEO_GEO(r_E, gm, h_GEO)
    LEOs = LG.walkerConstellation(h_LEO, inclination, numSat, numPlanes, phasing, name = "Sat")
    orbital_speed = LG.motion(r_E+h_LEO, 'meter/sec')/1000


    ##################### Seperate the satellites in each plane ##########################
    LEO_sats_in_plane = []
    for i in range(numPlanes):
        a = int(numSat/numPlanes)
        LEO_sats_in_plane.append(LEOs[i*a: a + i * a])


    ############################start and end time of satellites simulation#################################

    ts = sf.load.timescale()
    time1 = ts.utc(2022,9,22,00,00,00)      # Start point, UTC time
    time2 = ts.utc(2022,9,22,18,00,00)      # End point, UTC time
    seconds_difference_time = LG.difference_time_in_seconds(time1, time2) # How many seconds between start and end

    ####################################*********** Create the constellation and calculate the repsected parameters **************##########################################

    groundstation = wgs84.latlon(53.110987, 8.851239) #GS-BR
    GS_distance_from_Earth_center = LA.norm(groundstation.itrs_xyz.km)
    DF = LG.simulateConstellation(LEOs, groundstation, 30, time1, time2, ts = None, safetyMargin = 0)
    DF2 = DF.reset_index()
    # create an empty DataFrame to store satellite position data
    sat_position_df = pd.DataFrame(columns=['Satellite ID', 'Time', 'Sat Position (km)', 'GS Position (km)', 'Distance from Earth Surface (km)', 'distance to GS (km)', 'Elevation Angle (degree)'])
        
    sat_arr = []
    visiting_start_GS_LEO = []
    visiting_end_GS_LEO = []
    distance_GS_LEOs = []
    #%%
    """ ################################################# Parameters intialization for UAV, drones and HAPs ################################################################ """
    
    ############################################## First assign the environemnt of propagation and frequecny ################################
    Air_object = Air('Suburban', f)
    ############ Uav and Drone #############:
    # Insert the paramerts velocity and radius
    velocity = 18.0 # in km/h
    radius = 2 # in km
    ### call the init function of UAv_trajectory class:
    UAV = Uav_trajectory(velocity, radius, time_interval=1)
    # calculate the angular velocity and calculate number of steps 
    angular_velocity_rad, number_step = UAV.get_values()
    # 
    shiftak0 = 0
    ########## HASP Tarjectory ###############
    # Based on paper High Altitude Platform Stations (HAPS): Architecture and System Performance; by Yunchou Xing∗ , Frank Hsieh† , Amitava Ghosh†, and Theodore S. Rappaport∗
    velocity_haps = 75.0 # in km/hour
    radius_haps = 6 # in km
    HAPS1 =  HAPS_trajectory(velocity_haps, radius_haps, time_interval=1)
    angular_velocity_rad_haps, number_step_haps = HAPS1.get_values()
    shiftak0_haps = 0
    #%% All Data Frames initializations except satellite positions:    
    Interference_HAP_GS_df = pd.DataFrame(columns=['Time', 'HAP Position (km)', 'GS Position (km)', 'distance to GS (km)', 'Elevation Angle (degree)'])
    
    Interference_uav3_GS_df = pd.DataFrame(columns=['Time', 'UAV Position (km)', 'GS Position (km)', 'distance to GS (km)', 'Elevation Angle (degree)'])
    
    for i in range(0,100):
        Sat_ID = DF2.iloc[i,0]
        Sat_ID_int = int(Sat_ID[3:])
        t_rise  = DF2.iloc[i,1]
        t_set = DF2.iloc[i,2]
        t_rise_now = ts.utc(t_rise.year, t_rise.month, t_rise.day, t_rise.hour, t_rise.minute, t_rise.second)
        t_set_now = ts.utc(t_set.year, t_set.month, t_set.day, t_set.hour, t_set.minute, t_set.second)
        visibility_sec = LG.difference_time_in_seconds(t_rise_now, t_set_now)
        # In oder to calculate in miliseconds you need to convert visibility_sec to miliseconds by: visibility_sec*1000 and then in 
        # datetime.timedelta(seconds=i1) write instead of seconds as microseconds = i1*1000 because it doesn't accept miliseconds 
        
        """ ########################### Second loop over visibility duration of each satellite ############################ """
        
        for i1 in range(0,visibility_sec):

            time_now = t_rise_now+datetime.timedelta(seconds=i1)
            
            position_LEO = LEOs[Sat_ID_int-1].at(time_now)
            LEO_velocity = LEOs[Sat_ID_int-1].at(time_now).velocity
            LEO_velocity_km_per_sec = LEO_velocity.km_per_s
            print('satellite global position vector is:', position_LEO.xyz.km)
            print('satellite velocity in global coordinate vector is:', LEO_velocity_km_per_sec)
            
            # Earth's gravitational parameter
            mu = 398600.4418  # km^3/s^2
            
            # Angular momentum vector
            h = np.cross(position_LEO.xyz.km, LEO_velocity_km_per_sec)
            h_norm = np.linalg.norm(h)
            
            # Inclination
            inc_angle = np.arccos(h[2] / h_norm)
            
            # Node vector
            n = np.array([-h[1], h[0], 0])
            n_norm = np.linalg.norm(n)
            
            # Right Ascension of Ascending Node (RAAN)
            Omega = np.arctan2(n[1], n[0])
            
            # Eccentricity vector
            r_norm = np.linalg.norm(position_LEO.xyz.km)
            e = (np.cross(LEO_velocity_km_per_sec, h) / mu) - (position_LEO.xyz.km / r_norm)
            e_norm = np.linalg.norm(e)
            
            # Argument of Perigee
            omega = np.arccos(np.dot(n, e) / (n_norm * e_norm))
            if e[2] < 0:
                omega = 2 * np.pi - omega
            
            # Print results
            print(f"Inclination (inc_angle): {np.degrees(inc_angle):.2f} degrees")
            print(f"RAAN (Omega): {np.degrees(Omega):.2f} degrees")
            print(f"Argument of Perigee (omega): {np.degrees(omega):.2f} degrees")
            # Precompute trigonometric values
            cos_Omega = np.cos(Omega)
            sin_Omega = np.sin(Omega)
            cos_i = np.cos(inc_angle)
            sin_i = np.sin(inc_angle)
            cos_omega = np.cos(omega)
            sin_omega = np.sin(omega)
            
            # Construct the rotation matrix
            R_global_to_local = np.array([
                [
                    cos_omega * cos_Omega - sin_omega * cos_i * sin_Omega,
                    -cos_omega * sin_Omega - sin_omega * cos_i * cos_Omega,
                    sin_omega * sin_i
                ],
                [
                    sin_omega * cos_Omega + cos_omega * cos_i * sin_Omega,
                    -sin_omega * sin_Omega + cos_omega * cos_i * cos_Omega,
                    -cos_omega * sin_i
                ],
                [
                    sin_i * sin_Omega,
                    sin_i * cos_Omega,
                    cos_i
                ]
            ])
            
            # Print the resulting rotation matrix
            print("R_global_to_local:")
            print(R_global_to_local)
            # Compute the transpose of the rotation matrix (inverse for orthogonal matrices)
            R_local_to_global = R_global_to_local.T
            
            # Transform the global vector to the local frame
            LEO_position_local = R_local_to_global @ position_LEO.xyz.km
            
            print("LEO position in Local coordinate frame vector:", LEO_position_local)
            
            r_vsat_LEO_global = groundstation.at(time_now).position.km - position_LEO.xyz.km
            r_vsat_LEO_local = R_local_to_global @ r_vsat_LEO_global
            
            r_vsat_LEO_local_norm = np.linalg.norm(r_vsat_LEO_local)
            r_x_local, r_y_local, r_z_local = r_vsat_LEO_local
            # Elevation angle
            Theta_el_sat_see_vsat = np.arcsin(r_z_local / r_vsat_LEO_local_norm)
            
            # Azimuth angle
            Theta_az_sat_see_vsat = np.arctan2(r_y_local, r_x_local)
            
            # Print results
            print(f"Relative position of VSAT to LEO in satellite local frame: {r_vsat_LEO_local}")
            print(f"Elevation angle (Theta_el): {np.degrees(Theta_el_sat_see_vsat):.2f} degrees")
            print(f"Azimuth angle (Theta_az): {np.degrees(Theta_az_sat_see_vsat):.2f} degrees")
            
            r_sat_vsat_global = position_LEO.xyz.km - groundstation.at(time_now).position.km
            # Step 2: Compute space angles
            r_sat_vsat_global_norm = np.linalg.norm(r_sat_vsat_global)
            r_x_global, r_y_global, r_z_global = r_sat_vsat_global
            
            # Elevation angle
            theta_el_vsat_see_sat = np.arcsin(r_z_global / r_sat_vsat_global_norm)
            
            # Azimuth angle
            theta_az_vsat_see_sat = np.arctan2(r_y_global, r_x_global)
            
            # Print results
            print(f"Relative position of LEO to VSAT in global frame: {r_sat_vsat_global}")
            print(f"Elevation angle (theta_el): {np.degrees(theta_el_vsat_see_sat):.2f} degrees")
            print(f"Azimuth angle (theta_az): {np.degrees(theta_az_vsat_see_sat):.2f} degrees")
            
            position_GS = groundstation.at(time_now).position.km
            
            #distance_GS_sat = LA.norm(groundstation.at(time_now).position.km, position_LEO.xyz.km)
            distance_GS_sat = LG.distance(groundstation.at(time_now).position.km, position_LEO.xyz.km)
            
            elevation_angle = LG.elevation_angel_calculator(position_LEO.xyz.km, groundstation.at(time_now).position.km)
            
            # Create a new DataFrame with the row data
            new_row_sat_position = pd.DataFrame({
                'Satellite ID': [Sat_ID],
                'Time': [time_now.utc_datetime()],
                'Sat Position (km)': [position_LEO.xyz.km],
                'GS Position (km)': [groundstation.at(time_now).position.km],
                'Distance from Earth Surface (km)': [LA.norm(position_LEO.xyz.km) - r_E/1000],
                'distance to GS (km)': [distance_GS_sat],
                'Elevation Angle (degree)': [elevation_angle]
            })
            
            # Concatenate the new row DataFrame with the existing DataFrame
            sat_position_df = pd.concat([sat_position_df, new_row_sat_position], ignore_index=True)
            
            shiftak = (i1 + shiftak0) % number_step
            step = shiftak + 1
            # now for HAPS
            shiftak_haps = (i1 + shiftak0_haps) % number_step_haps
            step_haps = shiftak_haps + 1
                    
                ####### uav3 is an interferere to Ground station #######
            uav3_position = UAV.simulate_circular_trajectory(position_GS + np.array([0, 0, 0.1]), 0.1, step, position_GS)
            new_row_uav3_Rx_power_df = pd.DataFrame({
                'Time': [time_now.utc_datetime()],  # Wrap the scalar in a list
                'UAV Position (km)': [uav3_position[:,:3]],
                'GS Position (km)': [position_GS],  # Wrap the scalar in a list
                'distance to GS (km)': [np.linalg.norm(position_GS - uav3_position[:,:3])-2],  # Calculate the scalar
                'Elevation Angle (degree)': [uav3_position[:,-1]],  # Wrap the scalar in a list
            })
            Interference_uav3_GS_df= pd.concat([Interference_uav3_GS_df, new_row_uav3_Rx_power_df], ignore_index=True)

            #uav5_position = UAV.simulate_circular_trajectory(position_GS + np.array([5, 5, 10]), 10, step, position_GS)
            uav5_position = HAPS1.simulate_circular_trajectory(position_GS + np.array([7, 3, 10]), 10, step_haps, position_GS)
            uav5_loc = uav5_position[0,:3].flatten(); uav5_elev = uav5_position[0,-1].flatten()
            distance_GS_HAP = LG.distance(uav5_position[:,:3].flatten(), position_GS) 

            new_row_P_Rx_HAP_to_GS_df = pd.DataFrame({'Time':[time_now.utc_datetime()], 'HAP Position (km)':[uav5_position[:,:3]] , 'GS Position (km)':[position_GS], 'distance to GS (km)': [distance_GS_HAP], 'Elevation Angle (degree)':uav5_elev })
            Interference_HAP_GS_df =  pd.concat([Interference_HAP_GS_df, new_row_P_Rx_HAP_to_GS_df], ignore_index=True)


