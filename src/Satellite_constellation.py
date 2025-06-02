#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:29:05 2024

@author: vakilifard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:24:59 2024

@author: vakilifard
"""

import numpy as np
from numpy import linalg as LA
import datetime
from sgp4.api import Satrec, WGS72
from skyfield.api import load, wgs84, Topos
import skyfield.api as sf
from GEO import LEO_GEO
from Satellite_comm_param import Satellite_communication_parameter
from RX_power_calc import Rx_power
from Air_objects_class import Air
from HAPS_trajec_class import HAPS_trajectory
from fading_channel_sim import FadingSimulation
from Air_2_Ground_fading_channel import Fading_channel_air_to_Ground 
from shadowing_temporally_correlated_AR import ShadowingFading
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
    numPlanes = 12
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
    satellite_EIRP_total = 10*np.log10((10**(satellite_EIRP_density/10))*max_Bandwidth_per_beam)
    A_z = 1*10**(-1) #Based on the figure 4 of the document ITU-R P.676-13 other values for 30 GHz is 0.2 and 5 GHz is 0.04 in dB
    G_max_Rx = 32 #dBi    
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
    noise_figure_db = 3 # in dB
    temperature_k = 290 # in Kelvin
    GS_distance_from_Earth_center = LA.norm(groundstation.itrs_xyz.km)
    DF = LG.simulateConstellation(LEOs, groundstation, 20, time1, time2, ts = None, safetyMargin = 0)
    DF2 = DF.reset_index()
    # create an empty DataFrame to store satellite position data
    sat_position_df = pd.DataFrame(columns=['Satellite ID', 'Time', 'Sat Position (km)', 'GS Position (km)', 'Distance from Earth Surface (km)', 'distance to GS (km)', 'Elevation Angle (degree)'])
        
    sat_arr = []
    visiting_start_GS_LEO = []
    visiting_end_GS_LEO = []
    distance_GS_LEOs = []
    
    #%%#  *** The main loop of the program to generate position and channel for each involving element; The reference is based on the satellite movements *** #############
     
   ########################### First loop over satellites based on their rise over the area ############################
         # the 45th emerging satellite is equal to the 1st emerging satellite, we consider up to then it is not possible to change the frequency band
     #for i in range(45):
         # for in whole of the constellation in 24 hours
    for i in range(len(DF2)):
         # for over 4 satellites in a group
     #for i in range(3,7):
    #for i in range(0,1):
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
             
             position_GS = groundstation.at(time_now).position.km
             
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
             print(i1)
         print(i)
#%%
DF2_csv_location = '/home/vakilifard/Documents/codes_result/saved_data_from_simulations/Walker_constellation_600km/DF2.csv'
Satellite_df_position_csv_location = '/home/vakilifard/Documents/codes_result/saved_data_from_simulations/Walker_constellation_600km/satellite_position_df.csv'
# Save the DataFrame to a CSV file
DF2.to_csv(DF2_csv_location, index=True)
sat_position_df.to_csv(Satellite_df_position_csv_location, index=True)