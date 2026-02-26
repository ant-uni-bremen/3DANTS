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
from RX_power_calc import Rx_power
from Air_objects_class import Air
from Uav_trajec_class import Uav_trajectory
from HAPS_trajec_class import HAPS_trajectory
from Terresterial_Object import terresterial_network
from Gaussian_Field import Guassian_Random_filed_generator
from fading_channel_sim import FadingSimulation
from scipy.stats import ncx2
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
    satellite_EIRP = satellite_Tx_max_Gain + satellite_EIRP_density
    A_z = 1*10**(-1) #Based on the figure 4 of the document ITU-R P.676-13 other values for 30 GHz is 0.2 and 5 GHz is 0.04 in dB
    G_max_Rx = 25 #dBi
    
    
    #  Channel parameters w.r.t the Shadowed Rician Fading channel model and the parametrs reported in paper Performance Analysis of Satellite Communication System Under the Shadowed-Rician Fading: A Stochastic Geometry Approach, for 3 possibile cases
    # Three different shadowed-Rician fading models are taken into consideration: frequent heavy shadowing (FHS) {b = 0.063, m = 0.739, Ω = 8.97e4} named as Heavy
    # average shadowing (AS) {b = 0.126, m = 10.1, Ω = 0.835} named as Average, and infrequent light shadowing (ILS) {b = 0.158, m = 19.4, Ω = 1.29} named as Light
    #b = 0.158; m = 20; omega = 1.3; 
    N = 1000
    channel = Rx_power().ShadowedRicianRandGen('Light', N)
    
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
    DF = LG.simulateConstellation(LEOs, groundstation, 20, time1, time2, ts = None, safetyMargin = 0)
    DF2 = DF.reset_index()
    # create an empty DataFrame to store satellite position data
    sat_position_df = pd.DataFrame(columns=['Satellite ID', 'Time', 'Sat Position (km)', 'GS Position (km)', 'Distance from Earth Surface (km)', 'distance to GS (km)', 'Elevation Angle (degree)'])
        
    sat_arr = []
    visiting_start_GS_LEO = []
    visiting_end_GS_LEO = []
    distance_GS_LEOs = []
    
    generation_intervals = 100 # seconds.
    number_samples = generation_intervals * 1000
    satellite_fading_channel = FadingSimulation(num_samples = number_samples, fs = 1000, K = 0, N = 64, h = h_LEO, Doppler_compensate = 'Yes') 
    
    
    M = 3 # number of waves
    M0 = 3
    alpha_n = np.array([2*np.pi*(n-0.5)/M for n in range(1,M0+1)])
    beta_n = np.random.uniform(-np.pi,np.pi,M) #equivalent to phi
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
    # The initial parameter related to the K-Rician and function of elevation angle for Air to Groun calculation 
    K0 = 1; K_pi_half = 5; # values are in dB convert them to linear
    rho_direct = np.sqrt(50) # This value is for UAV to UAV calculation
    # Antenna trasnmitter gain of Uavs to ground as an omnidirectional antenna:
    G_tx_uav_type1 = 8;  #dBi based on the paper Investigation of Air-to-Air Channel Characteristics and a UAV Specific Extension to the Rice Model Niklas Goddemeier and Christian Wietfeld
    G_tx_rx_uav_2_uav = 20 # For air 2 air communication consider they use an array of patch antennas with high directivity 
    # Transmit power of UAVs:
    P_tx_uav_type1 = 30
    P_tx_uav_type2 = 43 # both in dBm
    
    # Channel generation for Ground - Air and Air-Air based on rician fading channel:
    K_ground_air = 5
    omega_ground_air = 1.1
    N1 = 100
    ground_air_channel = (np.sqrt(K_ground_air/(1+K_ground_air)))*Rx_power().Rician_Gen(K_ground_air, omega_ground_air, N1)
    #
    K_air_air = Air_object.air2air_K_calculator(200, 10000, rho_direct)
    omega_air_air = rho_direct**2/K_air_air
    N2 = 100
    air_air_channel =(np.sqrt(K_air_air/(1+K_air_air)))*Rx_power().Rician_Gen(K_air_air, omega_air_air, N2)
    
    """ Data needed for generation of UAVs as PPP in the area and for UAV1 moves randomly"""
    # Shared Data:
    time_step_uav = 1 # it must be in secoond, so for milisecond it will be 0.001 and so on
    # UAV1 specific data:
    position_center_uav1_moves_rand = groundstation.itrs_xyz.km + np.array([3,3,0.2])
    # drones per BSs data:
    radius_from_Bss = 10
    
    #%%
    """ ################################################# Parameters intialization for Terresterial network and objects ######################################################## """
    ############################################## First assign the environemnt of propagation and frequecny ################################
    Terresterial = terresterial_network(f, 'Suburban')
    radius_Terresterial = 50/2 # The value is in km
    ################################### Deployemnet scenarios ###################################
    #***********
    #Generate UEs and BSs on the ground: Scenario 1 --> Three BSs service a group of UEs and UAV2, Scenario 2 --> One of the base stations randomly over each satellite interfere Ground Station
    """NOTE: We have 3 scenarios for simulationg UEs on the ground:
        1- UEs are one time generated out of all for loops, and will be kept the same during whole simulation for passing all satellites
        2- UEs are generated at for each satellite passing (inside the for loop over DF2), but are kept constant during pass of the satellite
        3- UEs are fix in case of number for pass of each satellite, and are dynamic during satellite movemnet"""
    # scenario 1_1: Teresterial objects UEs are distributed over the area as Poisson Point Process to be served by BaseStations 
    lamb = 10 # This the lambda of PPP
    radius_per_each_BS = 19
    G_rx_UE_ppp = 5 #based on the paper 5G Cellular User Equipment: From Theory to Practical Hardware Design YIMING HUO1, (Student Member, IEEE), XIAODAI DONG1, (Senior Member, IEEE), AND WEI XU2, (Senior Member, IEEE)
    
    sigma_interference = 3 # This value is to simulate the pathloss experience from BaseStation as reference point to UEs around it
    ## scenario 1_2: Teresterial BaseStation are distributed over the area initially as a Binomial Point Process and then remained fixed 
    num_base_stations = 3
    [center_x, center_y, center_z] = groundstation.itrs_xyz.km
    #BaseStation_position0 = Terresterial.generate_base_station_positions(center_x, center_y, radius_Terresterial, num_base_stations, center_z)
    BaseStation_position0 = [np.array([3780,  603, 5077]), np.array([3780,  590 , 5077]), np.array([3795,  585, 5077])]
    BaseStation_position0_array = np.array(BaseStation_position0)
    BaseStation_latlon_coordinates = Terresterial.cartesian_to_latlon(np.array([BaseStation_position0]).reshape(-1,3))
    BaseStation_latlon_coordinates = np.array([BaseStation_latlon_coordinates]).reshape(-1,2)
    BaseStations_skyfield_positions = Terresterial.skyfield_position_for_BaseStations(BaseStation_latlon_coordinates)
    BS1 = BaseStations_skyfield_positions[0]
    BS2 = BaseStations_skyfield_positions[1] 
    BS3 = BaseStations_skyfield_positions[-1]
    height_BS = 35 # this value is in meter
    P_BSs_tx = 36 # in dBm
    G_BSs_tx = 8 + 10*np.log10(16) # dBi; This value is based on 3gpp TR 38.901 page 23 Table 7.3-1: Radiation power pattern of a single antenna element. The BS antenna is modelled by a uniform rectangular panel array, comprising MgNg panels
    
    """ UEs generation scenario1: UEs are one time generated and will be kept static during simulation. Make it comment for other scenarios
    """
    num_points = np.random.poisson(lamb)
    UEs_positions = Terresterial.generate_ue_clusters(np.array(BaseStation_position0), num_points,radius_per_each_BS)
    UEs_positions_array_2D = np.array(UEs_positions)[:, :, 0:2]
    UEs_latlon_coordinates = Terresterial.UEs_cartesian_to_latlon(np.array(UEs_positions))
    UEs_skyfield_positions = Terresterial.skyfield_position_for_UEs(UEs_latlon_coordinates)
    #%% Spatially Correlated Gaussian Random Field Generator for Shadowing
                        # **********************************************for Terrestrial links ************************************************************************ # 
    """Generate the Gaussian Random Field over area of coverage of each BaseStations in order to have a spatially corelated shadowing filed"""
    """No matter of UEs generation scenario, the filed is generated independent of UEs"""
    GaussianField = Guassian_Random_filed_generator()
    dimension_BS = 2
    variance_BS = 8
    len_sacel_terrestrial = 0.01 #in km: Based on Table-7.6.4.1-4: Spatial correlation distance for different scenarios from 3gpp TR 38.901 page 63
    field_BS1, srf_BS1 = GaussianField.field_generator_2D(dimension_BS, variance_BS, len_sacel_terrestrial, BaseStation_position0_array[0,0] , BaseStation_position0_array[0,1], radius_per_each_BS)
    #srf_BS1.vtk_export("field_BS1")
    ax = srf_BS1.plot()
    ax.set_aspect("equal")
    field_BS2, srf_BS2 = GaussianField.field_generator_2D(dimension_BS, variance_BS, len_sacel_terrestrial, BaseStation_position0_array[1,0] , BaseStation_position0_array[1,1], radius_per_each_BS)
    #srf_BS2.vtk_export("field_BS2")
    ax2 = srf_BS2.plot()
    ax2.set_aspect("equal")
    field_BS3, srf_BS3 = GaussianField.field_generator_2D(dimension_BS, variance_BS, len_sacel_terrestrial, BaseStation_position0_array[2,0] , BaseStation_position0_array[2,1], radius_per_each_BS)
    #srf_BS3.vtk_export("field_BS3")
    ax3 = srf_BS3.plot()
    ax3.set_aspect("equal")
                                                    #*************************************************************************#
    """Now for UEs generation scenario1: Based on the paper: Effects of Correlated Shadowing Modeling on Performance Evaluation of Wireless Sensor Networks by 
    Shani Lu1, John May, Russell J. Haines - formula 3 and 4. 
    So, for scenario1 of UEs generation: we measure the spatially correlated shadowing once for each UE per its BS and then use it during the rest of simulation. 
    For other UEs generation scenarios, need to figure it out"""
    #UEs_spatially_correlated_shadowing_per_BSs, value_of_shadowing_at_center = GaussianField.get_field_value_at_point(UEs_positions_array_2D, fields_of_all_BSs_array, BaseStation_position0_array[:,0:2], len_sacel_terrestrial)
    fields_of_all_BSs_array = np.column_stack((field_BS1, field_BS2, field_BS3))
    radius_of_movement = 1
    
                                                    #******************************* Small Scale Jake's Fading Channel ******************************************#
    
    """For generating the Jake's fading for UEs of BSs, we generate the parameters of vel_UEs, M_UEs_per_BS, M0_UEs, alpha_n_UEs_per_BS and doppler_UEs_per_BSs once and keep them same
    only for the phi_of_UEs_per_each_BS we gnerate in total in size of (M_UEs_per_BS, #BS, #UEs) and for each UE per each BS we feed M_UEs_per_BS to the program"""
    vel_UEs = 10
    M_UEs_per_BS = 8
    M0_UEs = 8
    alpha_n_UEs_per_BS = np.array([2*np.pi*(n-0.5)/M_UEs_per_BS for n in range(1,M0_UEs+1)])
    doppler_UEs_per_BSs = 2*np.pi*f*vel_UEs*np.cos(alpha_n_UEs_per_BS)/(3e8)
    ### generate the Phi_n of the Jake's fading model for all UEs per BaseStation and use them for generating the fading. The size must be (#base_station, #UEs)
    phi_of_UEs_per_each_BS = np.random.uniform(-np.pi,np.pi,(M_UEs_per_BS, 3, num_points))
    ######## For NTN connection
    phi_of_UEs_on_ground = np.random.uniform(-np.pi,np.pi,(M, 3, num_points))
    
    #%% All Data Frames initializations except satellite positions:
        # satellite to Ground station received power data frame
    P_Rx_data_set = pd.DataFrame(columns=['Satellite ID', 'Time', 'Elevation Angle (degree)','Distance (km)', 'LoS Prob (%)', 'P_Rx_fspl-1 (dBW)', 'P_Rx_fspl+shadow-2 (dBW)', 'P_Rx_fspl+Shadow-Rice-3 (dBW)', 'P_Rx_fspl+shadow+ShadowRice-4 (dBW)', 'P_Rx_fspl+shadow+Jake-5 (dBW)'])
    
    #uav1_Rx_power_df = pd.DataFrame(columns=['Time', 'UAV Position (km)', 'GS Position (km)', 'distance to GS (km)', 'Elevation Angle (degree)', 'P_rx (dBW)'])
    
    # Received power from BS1 by UAV2 dataframe
    uav2_Rx_power_fromBS1_df = pd.DataFrame(columns=['Time', 'UAV Position (km)', 'BS Position (km)', 'distance to BS (km)', 'Elevation Angle (degree)', 'P_rx (dBW)'])
    # The interference UAV3 creates in Ground station data frame
    Interference_uav3_GS_df = pd.DataFrame(columns=['Time', 'UAV Position (km)', 'GS Position (km)', 'distance to GS (km)', 'Elevation Angle (degree)', 'GS Antenna gain','P_rx (dBW)'])
    
    
    # The received power from HAP (UAV5) by UAV1 data frame
    HAP_to_uav1_Prx_df = pd.DataFrame(columns=['Time', 'UAV1 Position (km)', 'HAP Position (km)', 'distance (km)', 'P_rx (dBW)'])
    # Interference caused by Satellites to UAV1 data frame:
    Interference_Sat_uav1_df =  pd.DataFrame(columns=['Time', 'UAV Position (km)', 'Satellite Position (km)', 'distance (km)', 'Elevation Angle (degree)', 'P_rx (dBW)'])
    # Interference Satellites create on UAV2:
    Interference_Sat_uav2_df =  pd.DataFrame(columns=['Time', 'UAV Position (km)', 'Satellite Position (km)', 'distance (km)', 'Elevation Angle (degree)', 'P_rx (dBW)'])
    # Interference HAP causes on Ground station: 
    Interference_HAP_GS_df = pd.DataFrame(columns=['Time', 'HAP Position (km)', 'GS Position (km)', 'distance to GS (km)', 'Elevation Angle (degree)', 'GS Antenna gain (dB)','P_rx (dBW)'])
    #Interference HAP creates on uav2:
    Interference_HAP_uav2_df = pd.DataFrame(columns=['Time', 'UAV2 Position (km)', 'HAP Position (km)', 'distance (km)', 'Elevation Angle (Deg)','P_rx (dBW)'])
    
    # Received power at UEs from their serving BSs:
    UE_ppp_Rx_power_df = pd.DataFrame(columns=['Time', 'number of UE', 'P_rx (dBW)'])
    
    # Interference satellites create on UEs
    Interference_Sat_UEs_df = pd.DataFrame(columns=['Time', 'number of UE', 'Interference (dBW)'])
    
    # Interference HAP create on UEs
    Interference_HAP_UEs_df = pd.DataFrame(columns=['Time', 'number of UE', 'Interference (dBW)'])

    # Interference one of the Base stations on ground station:
    Interference_BaseStation_Gs_df = pd.DataFrame(columns=['Time', 'Base Station Number', 'Interference (dBW)'])
    #%% SIR and Rate dataframe creation

    Ground_Station_SIR_Rate = pd.DataFrame(columns=['Satellite ID', 'Time', 'Elevation Angle (degree)', 'SIR_HAP', 'Data Rate - Int-is-HAP (Mbit/sec)','SIR_uav3', 'Data Rate - Int-is-uav3 (Mbit/sec)','SIR_BS', 'Data Rate - Int-is-BS (Mbit/sec)','SIR_total', 'Data Rate - Int-is-Total (Mbit/sec)'])
    UAV1_SIR_Rate = pd.DataFrame(columns=['Time', 'Distance from HAP (km)', 'Satellite ID','SIR_satellite', 'Data Rate - Int-is-satellite (Mbit/sec)'])
    UAV2_SIR_Rate = pd.DataFrame(columns=['Time', 'Distance from BS1 (km)', 'Satellite ID','SIR_satellite', 'Data Rate - Int-is-satellite (Mbit/sec)', 'SIR_HAP', 'Data Rate - Int-is-HAP (Mbit/sec)', 'SIR_Total', 'Data Rate - Int-is-total (Mbit/sec)'])
    Satellite_to_UEs_Prx_SNR_df = pd.DataFrame(columns=['Time', 'Satellite ID', 'Elevation Angle (degree)', 'P_rx_total (dBW)' , 'SNR (dB)'])
    HAPS_to_UEs_Prx_SNR_df = pd.DataFrame(columns=['Time', 'Distance', 'Elevation Angle (degree)', 'P_rx_total (dBW)' , 'SNR (dB)'])
    GGG = []
   #%%
    """ ########################   *** The main loop of the program to generate position and channel for each involving element; The reference is based on the satellite movements *** #############"""
    
    """ ########################### First loop over satellites based on their rise over the area ############################ """
        # the 45th emerging satellite is equal to the 1st emerging satellite, we consider up to then it is not possible to change the frequency band
    #for i in range(45):
        # for in whole of the constellation in 24 hours
    #for i in range(len(DF2)):
        # for over 4 satellites in a group
    #for i in range(3,7):
    for i in range(0,10):
        Sat_ID = DF2.iloc[i,0]
        Sat_ID_int = int(Sat_ID[3:])
        t_rise  = DF2.iloc[i,1]
        t_set = DF2.iloc[i,2]
        t_rise_now = ts.utc(t_rise.year, t_rise.month, t_rise.day, t_rise.hour, t_rise.minute, t_rise.second)
        t_set_now = ts.utc(t_set.year, t_set.month, t_set.day, t_set.hour, t_set.minute, t_set.second)
        visibility_sec = LG.difference_time_in_seconds(t_rise_now, t_set_now)
        # In oder to calculate in miliseconds you need to convert visibility_sec to miliseconds by: visibility_sec*1000 and then in 
        # datetime.timedelta(seconds=i1) write instead of seconds as microseconds = i1*1000 because it doesn't accept miliseconds 
        
        """
        UEs generation scenario 2: UEs are generated for each new satellite and will be kept static during staellite pass. Make it comment for other scenarios:
        num_points = np.random.poisson(lamb)
        UEs_positions = Terresterial.generate_ue_clusters(np.array(BaseStation_position0), np.random.poisson(lamb),radius_per_each_BS)
        UEs_latlon_coordinates = Terresterial.UEs_cartesian_to_latlon(np.array(UEs_positions))
        UEs_skyfield_positions = Terresterial.skyfield_position_for_UEs(UEs_latlon_coordinates)
        """
        """ 
        UEs generation scenario 3: UEs are genearted for each new satellite pass and they are moving inside the BSs radius cluster during pass of satellite. 
        Make it comment for other scanrios:
        num_points = np.random.poisson(lamb)
        """
        # The new idea of making UEs moving randomly inside BSs area of coverage, by initially generate them as PPP and then make them moving randomly in a circle of radius 1 km
        UEs_positions_array_2D_new_each_i1 = UEs_positions_array_2D
        
        HAP_UEs_distance = np.zeros((visibility_sec, num_points))
        
        # Select one of the BaseStations which interfere the ground station per each satellite
        random_number_index = np.random.randint(0, 2)
        
        """ ########################### Second loop over visibility duration of each satellite ############################ """
        
        for i1 in range(0,visibility_sec*1000):

            time_now = t_rise_now+datetime.timedelta(microseconds=i1*1000)
            
            position_LEO = LEOs[Sat_ID_int-1].at(time_now)
            
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

            # Assign the propagation environemnt for line of sight probability claculating 
            lOS_prob = Rx_power().LOS_prob_calc(elevation_angle, 'Sub_Urban')
            
            # Calculate the channel gain from Shadowed Rician channel model 
            channel_gain_RicianShadowed_model = np.random.choice(a=channel, size=1, replace='True')
            
            # Calculate the channel gain based on Jake's fading spectrum 
            vel = 7500*(r_E*1000/(r_E*1000 + h_LEO*1000))*np.cos(elevation_angle)
            doppler = 2*np.pi*f*vel*np.cos(alpha_n)/(3e8)
            Channel_based_on_Jakes_fading = Rx_power().small_scale_fading(1, M, alpha_n, beta_n, doppler)
            
            # calculate the Shadowed-Rician fading channel temporally correlated as:
                # we generate the channel for each 100 seconds since the chnage of elevation angle each 100 seconds is significant
                # we generate per each second 1000 samples means TTi is 1 ms
            if i1 % generation_intervals == 0:
                print(f"Generating samples at i={i}, i1={i1}")
                shadowed_rician_channel_samples_interval = satellite_fading_channel.run_simulation(elevation_angle, f)
        
            start_idx = (i1 % generation_intervals) * 1000
            end_idx = start_idx + 1000
        
            if end_idx <= len(shadowed_rician_channel_samples_interval):
                Channel_fading_samples = shadowed_rician_channel_samples_interval[start_idx:end_idx]
            else:
                print(f"Index out of range at i={i}, i1={i1}")

            # calculate fspl and shadowing both based on the 3gpp tr 38.811 
            fspl_shadow_fading = Rx_power().Path_loss(f, distance_GS_sat, elevation_angle, 'LOS', 'SBand')
            
            # calculate only the fspl as function of distance and frequecny
            fspl_solo = Rx_power().FSPl_only(f, distance_GS_sat)
            
            atmospheric_loss = Rx_power().atmospheric_att(A_z, elevation_angle)
            
            # Here we calculate the received power for 5 cases: 
            # 1. only pathloss and atmospheric loss 
            # 2. pathloss and shadowing and atmospheric loss 
            # 3. only pathloss and channel gain of Shadowed rician model (no shadowing) 
            # 4. pathloss + shadowing + channel gain from RicianShadowed channel + atmospheric loss
            # 5. pathloss + shadowing + channel gain from Jake's model
            
            # case 1:
            P_received_fspl_case1 = (satellite_EIRP + G_max_Rx ) - fspl_solo - atmospheric_loss
            
            # case2:
            P_received_fspl_shadow_atm_case2 = (satellite_EIRP + G_max_Rx ) - fspl_shadow_fading - atmospheric_loss
            
            # case3: 
            P_received_fspl_ShadowRice_case3 = (satellite_EIRP + G_max_Rx + 20*np.log10(channel_gain_RicianShadowed_model)) - fspl_solo - atmospheric_loss
            
            # case4:
            P_received_fspl_ShadowRice_case4 = (satellite_EIRP + G_max_Rx + 20*np.log10(channel_gain_RicianShadowed_model)) - fspl_shadow_fading - atmospheric_loss
            
            # case5:
            P_received_fspl_ShadowRice_case5 = (satellite_EIRP + G_max_Rx + 20*np.log10(Channel_based_on_Jakes_fading)) - fspl_shadow_fading - atmospheric_loss
            #P_received_fspl_ShadowRice_case5 = (satellite_EIRP + G_max_Rx + 20*np.log10(np.abs(Channel_fading_samples))) - fspl_solo - atmospheric_loss
            
            new_row_P_Rx_data_set = pd.DataFrame({
                'Satellite ID': [Sat_ID],
                'Time': [time_now.utc_datetime()],
                'Elevation Angle (degree)': [elevation_angle],
                'Distance (km)': [distance_GS_sat],
                'LoS Prob (%)': [lOS_prob],
                'P_Rx_fspl-1 (dBW)': [P_received_fspl_case1],
                'P_Rx_fspl+shadow-2 (dBW)':  [P_received_fspl_shadow_atm_case2],
                'P_Rx_fspl+Shadow-Rice-3 (dBW)': [P_received_fspl_ShadowRice_case3],
                'P_Rx_fspl+shadow+ShadowRice-4 (dBW)': [P_received_fspl_ShadowRice_case4],
                'P_Rx_fspl+shadow+Jake-5 (dBW)': [list(P_received_fspl_ShadowRice_case5)]
            })
            P_Rx_data_set = pd.concat([P_Rx_data_set, new_row_P_Rx_data_set], ignore_index=True)
            
            """############### UAvs position and related things ####################:"""
            
            shiftak = (i1 + shiftak0) % number_step
            step = shiftak + 1
            # now for HAPS
            shiftak_haps = (i1 + shiftak0_haps) % number_step_haps
            step_haps = shiftak_haps + 1
            
                         # ***** pass GS/UE/BaseStation position in x y z and h and step tp uav_circular_trajectorty to calculate the interference  causing on the Ground Station***** #:
                
                ####### uav1 is served by HAP and got interfered by satellites #######
                # Here uav1 is moving in a deterministic path. We use it to generate the center position for uav1 random movement 
            uav1_position = UAV.simulate_circular_trajectory(position_GS + np.array([3,3,0.2]), 0.2, step, position_GS)
            uav1_loc_main_trajec = uav1_position[0,:3].flatten(); uav1_elev_main_trajec = uav1_position[0,-1].flatten()

            
                ####### ****** uav2 is getting service from BS1 always  ******** #######
            uav2_position = UAV.simulate_circular_trajectory(BS1.at(time_now).position.km+np.array([0, -5, 0.1]), 0.1, step, BS1.at(time_now).position.km)
            uav2_LoS_prob = Air_object.LoS_calculator(uav2_position[:,-1])
            uav2_pl_los, uav2_pl_nlos = Air_object.general_pathloss_calculator(BS1.at(time_now).position.km+np.array([0, 0, 0.035]), uav2_position[:,:3])
            uav2_channel_gain = Air_object.Rician_factor_calculator(uav2_position[:,-1], K0, K_pi_half)
            #uav2_channel_gain = np.random.choice(a=ground_air_channel, size=1, replace='True')
            rata = i1 // 50
            if rata % 2 == 0:
                uav2_BS1_antenna_tx_gain = G_BSs_tx
            else:
                uav2_BS1_antenna_tx_gain = 8
            GGG.append(uav2_BS1_antenna_tx_gain)
            uav2_Rx_power_from_BS1 = P_BSs_tx -30 + G_tx_uav_type1 + uav2_BS1_antenna_tx_gain + 10*np.log10(uav2_channel_gain) - (uav2_pl_los) # Here is no shadowing 
            new_row_uav2_Rx_power_df = pd.DataFrame({
                'Time': [time_now.utc_datetime()],  # Wrap the scalar in a list
                'UAV Position (km)': [uav2_position[:,:3]],
                'BS Position (km)': [BS1.at(time_now).position.km],  # Wrap the scalar in a list
                'distance to BS (km)': [np.linalg.norm(BS1.at(time_now).position.km - uav2_position[:,:3])],  # Calculate the scalar
                'Elevation Angle (degree)': [uav2_position[:,-1]],  # Wrap the scalar in a list
                'P_rx (dBW)': [uav2_Rx_power_from_BS1] # Wrap the scalar in a list
            })
            uav2_Rx_power_fromBS1_df = pd.concat([uav2_Rx_power_fromBS1_df, new_row_uav2_Rx_power_df], ignore_index=True)
            
            ######################### UAV2 is under interference from Sat #####################
            distance_uav2_sat = LG.distance(uav2_position[:,:3].flatten(), position_LEO.xyz.km) 
            elev_ang_uav2sat = LG.elevation_angel_calculator(position_LEO.xyz.km, uav2_position[:,:3].flatten())
            # we consider the same Rician Shadow fading channel for Satellite to UAV communication
            channel_sat2uav2_RicianShadowed = np.random.choice(a=channel, size=1, replace='True')
            # We only consider the free space pathloss from satelliet to uav no shadowing
            fspl_sat2uav2 = Rx_power().FSPl_only(f, distance_uav2_sat)   
            atmospheric_loss_Sat2Uav2 = Rx_power().atmospheric_att(A_z, elev_ang_uav2sat)
            P_rx_Sat_uav2 = (satellite_EIRP + G_tx_uav_type1 + 10*np.log10(channel_sat2uav2_RicianShadowed)) - fspl_sat2uav2 - atmospheric_loss_Sat2Uav2
            new_row_P_Rx_Sat_uav2_df = pd.DataFrame({'Time':[time_now.utc_datetime()], 'UAV Position (km)':[uav2_position[:,:3]] , 'Satellite Position (km)':[position_LEO.xyz.km], 'distance (km)': [distance_uav2_sat], 'Elevation Angle (degree)':elev_ang_uav2sat,'P_rx (dBW)':[ P_rx_Sat_uav2] })
            Interference_Sat_uav2_df =  pd.concat([Interference_Sat_uav2_df, new_row_P_Rx_Sat_uav2_df], ignore_index=True)
            
        
                ####### uav3 is an interferere to Ground station #######
            uav3_position = UAV.simulate_circular_trajectory(position_GS + np.array([0, 0, 0.1]), 0.1, step, position_GS)
            uav3_LoS_prob = Air_object.LoS_calculator(uav3_position[:,-1])
            UAV3_to_GS_angle = Rx_power().calculate_angle_interf(position_GS, position_LEO.xyz.km, uav3_position[:,:3].flatten())
            GS_antenna_gain_to_UAV3 = Rx_power().antenna_gain_calc( UAV3_to_GS_angle)
            uav3_pl_los, uav3_pl_nlos = Air_object.general_pathloss_calculator(position_GS, uav3_position[:,:3])
            #uav3_channel_gain = Air_object.Rician_factor_calculator(uav3_position[:,-1], K0, K_pi_half)
            uav3_channel_gain = np.random.choice(a=ground_air_channel, size=1, replace='True')
            sf_los3, sf_nlos3, cl_nlos3 = Air_object.Shadow_fading_Air_to_Ground(UAV3_to_GS_angle, 'LoS','SBand')
            
            # By thw way the antenna gain is calculated and when it is negative means that UAV3 in behind the GS so there is no receive! 
            if GS_antenna_gain_to_UAV3 >= 0:
                uav3_Rx_power = P_tx_uav_type1 -30 + G_tx_uav_type1 + GS_antenna_gain_to_UAV3 + 10*np.log10(uav3_channel_gain) - (uav3_pl_los+sf_los3)
                if np.isnan(uav3_Rx_power):
                    raise ValueError("A value has become NaN. Stopping the code for inspection.")
            else:
                uav3_Rx_power = -110
            new_row_uav3_Rx_power_df = pd.DataFrame({
                'Time': [time_now.utc_datetime()],  # Wrap the scalar in a list
                'UAV Position (km)': [uav3_position[:,:3]],
                'GS Position (km)': [position_GS],  # Wrap the scalar in a list
                'distance to GS (km)': [np.linalg.norm(position_GS - uav3_position[:,:3])-2],  # Calculate the scalar
                'Elevation Angle (degree)': [uav3_position[:,-1]],  # Wrap the scalar in a list
                'GS Antenna gain': GS_antenna_gain_to_UAV3,
                'P_rx (dBW)': [uav3_Rx_power] # Wrap the scalar in a list
            })
            Interference_uav3_GS_df= pd.concat([Interference_uav3_GS_df, new_row_uav3_Rx_power_df], ignore_index=True)
            
            
                ####### uav5 as HAP #######: *******************This UAV is about for Air-2-Air communication with one of the low altitudes uavs and cause interference on GS and UEs of BSs due to its higher radius **********************************
            #uav5_position = UAV.simulate_circular_trajectory(position_GS + np.array([5, 5, 10]), 10, step, position_GS)
            uav5_position = HAPS1.simulate_circular_trajectory(position_GS + np.array([7, 3, 20]), 20, step_haps, position_GS)
            uav5_loc = uav5_position[0,:3].flatten(); uav5_elev = uav5_position[0,-1].flatten()
            ##################  uav1 is getting service from HAPS, it is a Air2Air communication. uav1 can move random like here as a CIM model:
            uav1_position = UAV.UAV_trajectory_CIM(uav1_loc_main_trajec, 7.2, time_step_uav, uav5_loc, position_GS)
            uav1_loc = uav1_position[0,:3]; uav1_elev = uav1_position[0,-1]
            #position_center_uav1_moves_rand = uav1_loc
            
            # Here we consider that the LoS probability is 100% 
            #uav5_pl_shadow_los = Air_object.Air2Air_pathloss(uav1_position[:,:3], uav5_position[:,:3])
            uav5_pl_shadow_los = Rx_power().FSPl_only(f, np.linalg.norm(uav5_loc - uav1_loc))
            #HAP_uav1_air2air_channel =Air_object.air2air_Rician_channel_calculator(200, 10000, rho_direct)
            HAP_uav1_air2air_channel = np.random.choice(a=air_air_channel, size=1, replace='True')
            #uav5_Rx_power = P_tx_uav_type2 -30 + G_tx_rx_uav_2_uav + G_tx_uav_type1+2 - uav5_pl_shadow_los + 10*np.log10(HAP_uav1_air2air_channel)
            uav5_Rx_power = P_tx_uav_type2 -30 + G_tx_rx_uav_2_uav + G_tx_uav_type1 - uav5_pl_shadow_los
            new_row_uav5_Rx_power_df = pd.DataFrame({'Time':[time_now.utc_datetime()], 'UAV1 Position (km)':[uav1_position[:,:3]] , 'HAP Position (km)':[uav5_position[:,:3]], 'distance (km)': [np.linalg.norm(uav5_position[:,:3] - uav1_position[:,:3])], 'P_rx (dBW)':[uav5_Rx_power] })
            HAP_to_uav1_Prx_df = pd.concat([HAP_to_uav1_Prx_df, new_row_uav5_Rx_power_df], ignore_index=True)
            
            ################### Calculate the interference caused by Satelite communication to GS on uav1 which is getting service from uav5  #####################:
            distance_uav_sat = LG.distance(uav1_position[:,:3].flatten(), position_LEO.xyz.km) 
            elev_ang_uav1sat = LG.elevation_angel_calculator(position_LEO.xyz.km, uav1_position[:,:3].flatten())
            # we consider the same Rician Shadow fading channel for Satellite to UAV communication
            channel_sat2uav_RicianShadowed = np.random.choice(a=channel, size=1, replace='True')
            # We only consider the free space pathloss from satelliet to uav
            fspl_sat2uav = Rx_power().FSPl_only(f, distance_uav_sat)   
            atmospheric_loss_Sat2Uav = Rx_power().atmospheric_att(A_z, elev_ang_uav1sat)
            #P_rx_Sat_uav = (satellite_EIRP + G_tx_uav_type1 + 10*np.log10(channel_sat2uav_RicianShadowed)) - fspl_sat2uav - atmospheric_loss_Sat2Uav
            P_rx_Sat_uav = (satellite_EIRP + G_tx_uav_type1) - fspl_sat2uav - atmospheric_loss_Sat2Uav
            new_row_P_Rx_Sat_uav_df = pd.DataFrame({'Time':[time_now.utc_datetime()], 'UAV Position (km)':[uav1_position[:,:3]] , 'Satellite Position (km)':[position_LEO.xyz.km], 'distance (km)': [distance_uav_sat], 'Elevation Angle (degree)':elev_ang_uav2sat,'P_rx (dBW)':[ P_rx_Sat_uav] })
            Interference_Sat_uav1_df =  pd.concat([Interference_Sat_uav1_df, new_row_P_Rx_Sat_uav_df], ignore_index=True)
            
            ############################################# HAP (UAV5) interferer the Ground Station also: Since it has high altitude the gain of antenna w.r.t the coming signals from it can not be considered zero again there for we calculate as below:
                # If the angel of interference from HAP to GS w.r.t satellite (since GS antenna main beam is locked on satellite) causes a gain zero or less than zero it will be considered zero
            HAP_to_GS_angle = Rx_power().calculate_angle_interf(position_GS, position_LEO.xyz.km, uav5_position[:,:3].flatten())
            GS_antenna_gain_to_HAP = Rx_power().antenna_gain_calc( HAP_to_GS_angle)
            distance_GS_HAP = LG.distance(uav5_position[:,:3].flatten(), position_GS) 
            channel_HAP_to_GS_RicianShadowed = np.random.choice(a=channel, size=1, replace='True')
            if GS_antenna_gain_to_HAP >= 0:
                P_Rx_HAP_to_GS = P_tx_uav_type2 -30 + G_tx_rx_uav_2_uav + Rx_power().HAP_interferer(distance_GS_HAP, GS_antenna_gain_to_HAP, channel_HAP_to_GS_RicianShadowed, HAP_to_GS_angle, f)
                if np.isnan(P_Rx_HAP_to_GS):
                    raise ValueError("A value has become NaN. Stopping the code for inspection.")
            else:
                P_Rx_HAP_to_GS = -110
            new_row_P_Rx_HAP_to_GS_df = pd.DataFrame({'Time':[time_now.utc_datetime()], 'HAP Position (km)':[uav5_position[:,:3]] , 'GS Position (km)':[position_GS], 'distance to GS (km)': [distance_GS_HAP], 'Elevation Angle (degree)':HAP_to_GS_angle, 'GS Antenna gain (dB)': [GS_antenna_gain_to_HAP],'P_rx (dBW)':[ P_Rx_HAP_to_GS] })
            Interference_HAP_GS_df =  pd.concat([Interference_HAP_GS_df, new_row_P_Rx_HAP_to_GS_df], ignore_index=True)
            
            ###################### UAV2 is under interference from HAP #################
            distance_uav2_HAP = LG.distance(uav2_position[:,:3].flatten(), uav5_position[:,:3].flatten()) 
            elev_ang_uav2_HAP = LG.elevation_angel_calculator(uav5_position[:,:3].flatten(), uav2_position[:,:3].flatten())
            # we consider since uav2 is close to the earth, it experience shadowing and small scale fading like a NTN connection
            # Here we consider that the LoS probability is 100% 
            HAP_to_uav2_pathloss = Air_object.Air2Air_pathloss(uav2_position[:,:3], uav5_position[:,:3])
            HAP_to_uav2_shadowing_los = np.array(Air_object.Shadow_fading_Air_to_Ground(elev_ang_uav2_HAP, 'LoS', 'SBand'))[0]
            HAP_to_uav2_channel_gain = np.random.choice(a=air_air_channel, size=1, replace='True')
            #HAP_to_uav2_Interference_power = P_tx_uav_type2 -30 + G_tx_rx_uav_2_uav + G_tx_uav_type1 - HAP_to_uav2_pathloss + 10*np.log10(np.abs(Air_object.air2air_Rician_channel_calculator(100, 10000, rho_direct)))
            HAP_to_uav2_Interference_power = P_tx_uav_type2 -30 + G_tx_rx_uav_2_uav + G_tx_uav_type1+ 10*np.log10(HAP_to_uav2_channel_gain) - HAP_to_uav2_pathloss 
            new_row_Interference_HAP_uav2_df = pd.DataFrame({'Time':[time_now.utc_datetime()], 'UAV2 Position (km)':[uav2_position[:,:3]] , 'HAP Position (km)':[uav5_position[:,:3]], 'distance (km)': [np.linalg.norm(uav5_position[:,:3] - uav2_position[:,:3])], 'Elevation Angle (Deg)':elev_ang_uav2_HAP ,'P_rx (dBW)':[HAP_to_uav2_Interference_power] })
            Interference_HAP_uav2_df = pd.concat([Interference_HAP_uav2_df, new_row_Interference_HAP_uav2_df], ignore_index=True)

            
                                                    ###################### terresterial objects #############################:
                
            ### convert BaseStation positions to correct x,y,z in km based on time_now, the output is list
            BaseStation_positions_time_now = Terresterial.get_base_station_positions_at_time_now(time_now, BaseStations_skyfield_positions, height_BS)
            
            """ 
            UEs generation scenario 1 or 2: Get the UEs positions at time now, Part 1-2 and Part 2-2:
            NOTE: This is for the case that we consider that UEs are not moving during pass of a satellite!
            """
            UEs_positions_time_now = Terresterial.get_UEs_positions_at_time_now(time_now, UEs_skyfield_positions, 1.5)
            UEs_positions_time_now_array = np.array(UEs_positions_time_now)
            UEs_positions_time_now_array_reshaped = UEs_positions_time_now_array.reshape(num_base_stations,num_points,3)
            
            """ UEs generation scenario 3: UEs are not static during satrellite pass, and they are moving: Part 3-2:
            UEs generation in *scenario 1* which BaseStation serving to UEs, the output is a list in shape of: (number of base stations, number of UEs per base sation, 3 which is x, y, z coordinates)
            It can be used as well to generate UAVs in the target area
            Make it comment for other scenarios
            """
            num_uavs = np.random.poisson((np.pi * 10**2 * 0.1 / 3) * 0.3)
            max_height = 0.3
            UAVs_positions_centered_by_BSs = Terresterial.generate_uavs_clusters(np.array(BaseStation_positions_time_now), num_uavs, radius_from_Bss, max_height)
            UAVs_positions_centered_by_BSs_array = np.array(UAVs_positions_centered_by_BSs)
            UAVs_asign_to_bs = Air_object.assign_uavs_to_bs(UAVs_positions_centered_by_BSs_array, np.array(BaseStation_positions_time_now))
            
            """In order to keep consistency in the code. We keep working with the parametr UE_positions_per_Bs_array"""
            UE_positions_per_Bs_array = UEs_positions_time_now_array_reshaped
            ########################## UEs generation is scenario 1:    
            # Generate the received power from BaseStations to UEs per each cluster. The frequency is only and only considered below 7 GHz here for BaseStations to UEs communication:
            # UE_positions_served_by_BSs is a list when convert to np.array an extra dimension will be created and will be for example like: (1, 3: Number of BSs, 10: Number of UEs per BS, 3: x y z)
            
            """An idea! change the position of UEs which are generated by Poisson point process inside a smaller circle randomly"""
            UEs_positions_array_2D_new_each_i1 = Terresterial.UEs_random_movement_inside_BS_area(UEs_positions_array_2D_new_each_i1, radius_of_movement)
            """Now feed the shadowing spatially correlated calculator each iteration over i1 a new position"""
            UEs_spatially_correlated_shadowing_per_BSs, value_of_shadowing_at_center = GaussianField.get_field_value_at_point(UEs_positions_array_2D_new_each_i1, fields_of_all_BSs_array, BaseStation_position0_array[:,0:2], len_sacel_terrestrial)
            
            aa , bb = Terresterial.pathloss_BaseStations_to_UEs(np.array(BaseStation_positions_time_now), UE_positions_per_Bs_array, UEs_spatially_correlated_shadowing_per_BSs,height_BS, 1.2)
            aa = aa.reshape(aa.shape[1], aa.shape[2]); bb = bb.reshape(bb.shape[1], bb.shape[2])
            
            # Now we add the channel to the arrays, for LoS it is Rician and for NLoS it is rayleigh
            UEs_BSs_terrestrial_channel = Terresterial.small_scale_Jake_fading(1, M_UEs_per_BS , alpha_n_UEs_per_BS , phi_of_UEs_per_each_BS, doppler_UEs_per_BSs, aa)
            #rayleigh_channel = np.full_like(aa, Rayleigh_channel_generation[i1])
            #UE_ppp_Rx_power = (np.full_like(aa, P_BSs_tx + G_BSs_tx + G_rx_UE_ppp)) - (aa+bb+ 20*np.log10(np.sqrt(1/100)*np.random.rayleigh(0.5, aa.shape))) 
            UE_ppp_Rx_power = (np.full_like(aa, P_BSs_tx + G_BSs_tx + G_rx_UE_ppp)) - (aa+bb+ 20*np.log10(UEs_BSs_terrestrial_channel))
            new_row_UE_ppp_Rx_power_df = pd.DataFrame({'Time': [time_now.utc_datetime()] , 'number of UE':[num_points], 'P_rx (dBW)':[UE_ppp_Rx_power]})
            UE_ppp_Rx_power_df = pd.concat([UE_ppp_Rx_power_df, new_row_UE_ppp_Rx_power_df], ignore_index=True)
            
                                                            #* Satellite Interference to UEs *#
            # Now we generate the Interference caused by the satellite to UEs, in such a way that the interefrence power received at 
            # BaseStations is calculated and then distributed over the area of 5 km radius around them by a circular Gaussian random variable 
            Sat_BSs_distances = np.linalg.norm(np.array(BaseStation_positions_time_now) - position_LEO.xyz.km, axis=1)
            #elevations_angles_BSs_sat = np.full(num_base_stations, 90) - Terresterial.elevation_angle_Sat_BSs(np.array(BaseStation_positions_time_now), position_LEO.xyz.km)
            #elevations_angles_BSs_sat = Terresterial.elevation_angle_Sat_BSs(np.array(BaseStation_positions_time_now), position_LEO.xyz.km)
            elevations_angles_BSs_sat = np.full(num_base_stations, elevation_angle)
            Sat_to_BSs_area_channel_RicianShadowed = np.random.choice(a=channel, size=(aa.reshape(3,num_points)).shape, replace='True')
            Sat_BSs_interference = Terresterial.Sat_to_Bs_Interference(Sat_BSs_distances, elevations_angles_BSs_sat, A_z, satellite_EIRP)
            Satellite_to_UEs_interference = Terresterial.Satellite_to_UE_interefernce(Sat_BSs_interference, elevations_angles_BSs_sat, aa, num_points)
            new_row_P_Rx_Sat_UEs_df = pd.DataFrame({'Time': [time_now.utc_datetime()] , 'number of UE':[num_points], 'Interference (dBW)': [Satellite_to_UEs_interference]})
            Interference_Sat_UEs_df = pd.concat([Interference_Sat_UEs_df, new_row_P_Rx_Sat_UEs_df], ignore_index=True)
            
            # Scenario 2 : one of the BaseStations randomly per each satellite interferers the Ground Station
            BaseStation_positions_array = np.array(BaseStation_positions_time_now)
            Interferer_BS_to_GS = BaseStation_positions_array[random_number_index]
            BaseStation_to_GS_interference = (P_BSs_tx + G_BSs_tx + 0) + Terresterial.BaseStation_to_GroundStation_Interfeerence(Interferer_BS_to_GS, position_GS, height_BS, 5)
            new_row_BaseStation_Interferer_Gs = pd.DataFrame({'Time':[time_now.utc_datetime()], 'Base Station Number':[random_number_index], 'Interference (dBW)':[BaseStation_to_GS_interference]})
            Interference_BaseStation_Gs_df = pd.concat([Interference_BaseStation_Gs_df, new_row_BaseStation_Interferer_Gs], ignore_index=True)
            
            ### HAP interferers the UEs of BaseStations, but BS3 is out of it's coverage of beam diameter on the ground. 
            # HAP antenna based on 3gpp TR 38.811 page 46 can be considred as a a uniform rectangular panel array, which has beamwidth of theta_3dB = 2/N where N is the Number of antenna element. if N = 4 then theta_3dB is 0.5 rad ref: https://dsp.stackexchange.com/questions/39010/difference-between-uniform-linear-array-ula-3-db-beamwidth-and-bearing-resolut
            # Here we consider an omni-directional antenna for HAP with 3 dB beamwidth of 60 degree then the diameter on earth is 4618 meter with current height. the point is not all Bss are under coverage of HAP, so at each HAP position, project its position in 2D and see if the distnace with BS position 2D is more or less
            HAP_BSs_distances = np.linalg.norm(np.array(BaseStation_positions_time_now) - uav5_position[:,:3], axis=1)
            #### HAP distance to UEs; here we calculate only for BS1 as an example:
            HAP_UEs_d = np.transpose(np.linalg.norm(UE_positions_per_Bs_array[0,:,:] - uav5_position[:,:3], axis=1))
            HAP_UEs_distance[i1,:] = HAP_UEs_d.reshape(1,num_points)
            ####
            elevations_angles_BSs_HAP = Terresterial.elevation_angle_Sat_BSs(np.array(BaseStation_positions_time_now), uav5_position[:,:3].flatten())
            HAP_to_BSs_area_channel_RicianShadowed = np.random.choice(a=channel, size=(aa.reshape(3,num_points)).shape, replace='True')
            HAP_BSs_interference = Terresterial.Sat_to_Bs_Interference(HAP_BSs_distances, elevations_angles_BSs_HAP, A_z, P_tx_uav_type2+G_tx_rx_uav_2_uav-30) 
            HAP_to_UEs_interference = np.array(Terresterial.Satellite_to_UE_interefernce(HAP_BSs_interference, elevations_angles_BSs_HAP, aa, num_points))
            new_row_P_Rx_HAP_UEs_df = pd.DataFrame({'Time': [time_now.utc_datetime()] , 'number of UE':[num_points], 'Interference (dBW)': [HAP_to_UEs_interference]})
            Interference_HAP_UEs_df = pd.concat([Interference_HAP_UEs_df, new_row_P_Rx_HAP_UEs_df], ignore_index=True)
            
            
            """ UEs on the ground received power from Satellite, HAPS or UAV as their main service provider"""
            vel_Sat_UEs = 7500*(r_E*1000/(r_E*1000 + h_LEO*1000))*np.cos(elevations_angles_BSs_sat)
            doppler_Sat_UEs = Terresterial.doppler_Sat_UE(vel_Sat_UEs, alpha_n)
            Sat_UEs_ss_fading = Terresterial.Sat_UE_small_scale_Jakes(1, M , alpha_n , phi_of_UEs_on_ground, doppler_Sat_UEs, aa)
            Satellite_to_UEs_Prx = np.array(Satellite_to_UEs_interference) - 20*np.log10(Sat_UEs_ss_fading)
            #Satellite_to_UEs_Prx = np.array(Satellite_to_UEs_interference)
            new_row_Satellite_to_UEs_Prx_SNR_df = pd.DataFrame({'Time': [time_now.utc_datetime()], 'Satellite ID': [Sat_ID], 'Elevation Angle (degree)': [elevations_angles_BSs_sat], 'P_rx_total (dBW)': [Satellite_to_UEs_Prx] , 'SNR (dB)': [Satellite_to_UEs_Prx +120]})
            Satellite_to_UEs_Prx_SNR_df = pd.concat([Satellite_to_UEs_Prx_SNR_df, new_row_Satellite_to_UEs_Prx_SNR_df], ignore_index = True)
            # HAPS
            new_row_HAPS_to_UEs_Prx_SNR_df = pd.DataFrame({'Time': [time_now.utc_datetime()], 'Distances': [HAP_BSs_distances], 'Elevation Angle (degree)': [elevations_angles_BSs_HAP], 'P_rx_total (dBW)': [HAP_to_UEs_interference] , 'SNR (dB)': [HAP_to_UEs_interference +120]})
            HAPS_to_UEs_Prx_SNR_df = pd.concat([HAPS_to_UEs_Prx_SNR_df, new_row_HAPS_to_UEs_Prx_SNR_df], ignore_index = True)
            
            ############################################################# SIR DataFrame fullfill  ###########################################
            SIR_on_GS_HAP_is_Interferer = 10**((P_received_fspl_ShadowRice_case4 - P_Rx_HAP_to_GS)/10)
            SIR_on_GS_uav3_is_Interferer = 10**((P_received_fspl_ShadowRice_case4 - uav3_Rx_power)/10)
            SIR_on_GS_BaseStation_is_Interferer = 10**((P_received_fspl_ShadowRice_case4 - BaseStation_to_GS_interference)/10)
            SIR_on_GS_Total = 10**(P_received_fspl_ShadowRice_case4/10)/((10**(P_Rx_HAP_to_GS/10)) + (10**(uav3_Rx_power/10)) + (10**(BaseStation_to_GS_interference/10)))
            R_on_GS_HAP_is_interferer = max_Bandwidth_per_beam*np.log2(1 + SIR_on_GS_HAP_is_Interferer)
            R_on_GS_uav3_is_interferer = max_Bandwidth_per_beam*np.log2(1 + SIR_on_GS_uav3_is_Interferer)
            R_on_GS_Base_is_interferer = max_Bandwidth_per_beam*np.log2(1 + SIR_on_GS_BaseStation_is_Interferer)
            R_on_GS_Total = max_Bandwidth_per_beam*np.log2(1 + SIR_on_GS_Total)
            new_row_Ground_Station_SIR_Rate = pd.DataFrame({
                'Satellite ID': [Sat_ID],
                'Time': [time_now.utc_datetime()],
                'Elevation Angle (degree)': [elevation_angle],
                'SIR_HAP': SIR_on_GS_HAP_is_Interferer,
                'Data Rate - Int-is-HAP (Mbit/sec)': R_on_GS_HAP_is_interferer,
                'SIR_uav3': SIR_on_GS_uav3_is_Interferer,
                'Data Rate - Int-is-uav3 (Mbit/sec)': R_on_GS_uav3_is_interferer,
                'SIR_BS': SIR_on_GS_BaseStation_is_Interferer,
                'Data Rate - Int-is-BS (Mbit/sec)': R_on_GS_Base_is_interferer,
                'SIR_total': SIR_on_GS_Total,
                'Data Rate - Int-is-Total (Mbit/sec)': R_on_GS_Total
            })
            Ground_Station_SIR_Rate = pd.concat([Ground_Station_SIR_Rate, new_row_Ground_Station_SIR_Rate], ignore_index=True)
            
            SIR_on_uav1_Sat_is_Interferer = 10**((uav5_Rx_power - P_rx_Sat_uav)/10)
            R_on_uav1_Sat_is_Interferer  = max_Bandwidth_per_beam*np.log2(1 + SIR_on_uav1_Sat_is_Interferer)
            new_row_UAV1_SIR_Rate = pd.DataFrame({
                'Time': [time_now.utc_datetime()], 
                'Distance from HAP (km)': [np.linalg.norm(uav5_position[:,:3] - uav1_position[:,:3])], 
                'Satellite ID':[Sat_ID],
                'SIR_satellite': SIR_on_uav1_Sat_is_Interferer, 
                'Data Rate - Int-is-satellite (Mbit/sec)':R_on_uav1_Sat_is_Interferer
            })
            UAV1_SIR_Rate = pd.concat([UAV1_SIR_Rate, new_row_UAV1_SIR_Rate], ignore_index=True)
            
            SIR_on_UAV2_Sat_is_Interferer = 10**((uav2_Rx_power_from_BS1 - P_rx_Sat_uav2)/10)
            R_on_UAV2_Sat_is_Interferer = max_Bandwidth_per_beam*np.log2(1 + SIR_on_UAV2_Sat_is_Interferer)
            SIR_on_UAV2_HAP_is_Interferer = 10**((uav2_Rx_power_from_BS1 - HAP_to_uav2_Interference_power)/10)
            R_on_UAV2_HAP_is_Interferer = max_Bandwidth_per_beam*np.log2(1 + SIR_on_UAV2_HAP_is_Interferer)
            SIR_on_UAV2_Total = (10**(uav2_Rx_power_from_BS1/10))/((10**(P_rx_Sat_uav2/10))+(10**(HAP_to_uav2_Interference_power/10)))
            R_on_UAV2_Total = max_Bandwidth_per_beam*np.log2(1 + SIR_on_UAV2_Total)
            new_row_UAV2_SIR_Rate = pd.DataFrame({
                'Time': [time_now.utc_datetime()], 
                'Distance from BS1 (km)': [np.linalg.norm(BS1.at(time_now).position.km - uav2_position[:,:3])],
                'Satellite ID': [Sat_ID],
                'SIR_satellite': SIR_on_UAV2_Sat_is_Interferer,
                'Data Rate - Int-is-satellite (Mbit/sec)': R_on_UAV2_Sat_is_Interferer,
                'SIR_HAP': SIR_on_UAV2_HAP_is_Interferer,
                'Data Rate - Int-is-HAP (Mbit/sec)': R_on_UAV2_HAP_is_Interferer,
                'SIR_Total': SIR_on_UAV2_Total,
                'Data Rate - Int-is-total (Mbit/sec)': R_on_UAV2_Total
                })
            UAV2_SIR_Rate = pd.concat([UAV2_SIR_Rate, new_row_UAV2_SIR_Rate], ignore_index=True)

            print(i1)
        shiftak0 = (shiftak0 + visibility_sec) % number_step
        shiftak = (visibility_sec + shiftak0) % number_step
        #
        shiftak0_haps = (shiftak0_haps + visibility_sec) % number_step_haps
        shiftak_haps = (visibility_sec + shiftak0_haps) % number_step_haps
        print(i)
        # Create a 2D plot
        plt.figure(figsize=(8, 6))
        plt.scatter(position_GS[0], position_GS[1], label='Ground Station', c='red', marker='o')
        for iii, bs in enumerate(BaseStation_positions_time_now):
            plt.scatter(bs[0], bs[1], label=f'Base Station {iii + 1}', marker="d", s=150)
        colors = ['blue', 'orange', 'green']
        for iii1 in range(UE_positions_per_Bs_array.shape[0]):
            for jjj in range(UE_positions_per_Bs_array.shape[1]):
                ue = UE_positions_per_Bs_array[iii1,jjj,:]
                label = f'UE served by BS {iii1 + 1}' if jjj == 0 else "_nolegend_"
                plt.scatter(ue[0], ue[1], c=colors[iii1], marker='s',label=label)
            for jjj2 in range(UAVs_positions_centered_by_BSs_array.shape[1]):
                uav = UAVs_positions_centered_by_BSs_array[iii1, jjj2, :]
                label = f'UAV assigned to BS {iii1 + 1}' if jjj2 == 0 else "_nolegend_"
                plt.scatter(uav[0], uav[1], c=colors[iii1], marker='^',label=label)
        """for iii, bs_ue_list in enumerate(UE_positions_served_by_BSs):
            for jjj, ue in enumerate(bs_ue_list):
                label = f'UE served by BS {iii + 1}' if jjj == 0 else "_nolegend_"
                plt.scatter(ue[0], ue[1], c=colors[iii], marker='s',label=label)"""
        plt.scatter(uav5_position[:,0], uav5_position[:,1], label = 'HAP', c = 'm', marker = "h")
        plt.scatter(uav1_position[:,0], uav1_position[:,1], label = 'UAV1', c = 'k', marker = "^")
        plt.scatter(uav2_position[:,0], uav2_position[:,1], label = 'UAV2', c = 'y', marker = "^")
        plt.scatter(uav3_position[:,0], uav3_position[:,1], label = 'UAV3', c = 'c', marker = "^")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='upper right',bbox_to_anchor=(1.1, 1), prop={'size': 7})
        plt.grid()
        plt.show()
        ###############################################################################################################
        W = np.array(UE_ppp_Rx_power_df['P_rx (dBW)']) #auxilary variable
        UEs_per_BSs = np.array([w.reshape(num_base_stations, num_points) for w in W])
        UEs_BSs1_0 = UEs_per_BSs[:,0,:]
        UEs_BSs2_0 = UEs_per_BSs[:,1,:]
        UEs_BSs3_0 = UEs_per_BSs[:,2,:]
        S = np.array(Interference_Sat_UEs_df['Interference (dBW)']) #auxilary variable 
        Sat_on_UEs = np.array([np.array(w).reshape(num_base_stations, num_points) for w in S])
        UEs_of_BS1_Sat_Interferer_0 = Sat_on_UEs[:,0,:]
        UEs_of_BS2_Sat_Interferer_0 = Sat_on_UEs[:,1,:]
        UEs_of_BS3_Sat_Interferer_0 = Sat_on_UEs[:,2,:]
        H = np.array(Interference_HAP_UEs_df['Interference (dBW)']) #auxilary variable
        HAP_on_UEs = np.array([np.array(w).reshape(num_base_stations, num_points) for w in H])
        UEs_of_BS1_HAPS_Interferer_0 = HAP_on_UEs[:,0,:]
        UEs_of_BS2_HAPS_Interferer_0 = HAP_on_UEs[:,1,:]
        UEs_of_BS3_HAPS_Interferer_0 = HAP_on_UEs[:,2,:]
        SIR_UEs_BS1_Sate_and_HAP_0 = np.divide(10**(UEs_BSs1_0/10), (10**(UEs_of_BS1_Sat_Interferer_0/10))+(10**(UEs_of_BS1_HAPS_Interferer_0/10)))
        SIR_UEs_BS2_Sate_and_HAP_0 = np.divide(10**(UEs_BSs2_0/10), (10**(UEs_of_BS2_Sat_Interferer_0/10))+(10**(UEs_of_BS2_HAPS_Interferer_0/10)))
        SIR_UEs_BS3_Sate_0 = np.divide(10**(UEs_BSs3_0/10), (10**(UEs_of_BS3_Sat_Interferer_0/10)))
        #Satellite
        SINR_UEs_BS1_Sat = np.divide(10**(UEs_BSs1_0/10), 10**(UEs_of_BS1_Sat_Interferer_0/10)+10**(-120/10))
        SINR_UEs_BS2_Sat = np.divide(10**(UEs_BSs2_0/10), 10**(UEs_of_BS2_Sat_Interferer_0/10)+10**(-120/10))
        SINR_UEs_BS3_Sat = np.divide(10**(UEs_BSs3_0/10), 10**(UEs_of_BS3_Sat_Interferer_0/10)+10**(-120/10))
        #HAPS
        SINR_UEs_BS1_HAPS = np.divide(10**(UEs_BSs1_0/10), 10**(UEs_of_BS1_HAPS_Interferer_0/10)+10**(-120/10))
        SINR_UEs_BS2_HAPS = np.divide(10**(UEs_BSs2_0/10), 10**(UEs_of_BS2_HAPS_Interferer_0/10)+10**(-120/10))
        SINR_UEs_BS3_HAPS = np.divide(10**(UEs_BSs3_0/10), 10**(UEs_of_BS3_HAPS_Interferer_0/10)+10**(-120/10))


    satellites_df = sat_position_df
    satellites_df['Time'] = pd.to_datetime(satellites_df['Time'])
    satellites_df = satellites_df.sort_values('Time')
    
# %%
"""#####################################################************** Plot section ***************################################################## """
# Specify the directory where you want to save the plots
save_directory = '/home/vakilifard/Documents/My papers/AP4_Open_6GHub_paper/new-figures'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
# Plot auto-correlation function with self written function 
def auto_correlation(x, max_predict):
    auto_corr = np.zeros(max_predict)
    auto_corr[0] = np.corrcoef(x)
    for i in range(1, max_predict):
        auto_corr[i] = np.corrcoef(x[i:],x[:-i])[0,1]*(len(x)-i)/len(x)
    return auto_corr
# X axis and Y axis 
x1 = P_Rx_data_set['Distance (km)']
x2  = P_Rx_data_set['Elevation Angle (degree)']
y1 = P_Rx_data_set['P_Rx_fspl-1 (dBW)']
y2=  P_Rx_data_set['P_Rx_fspl+shadow-2 (dBW)']
y3 = P_Rx_data_set['P_Rx_fspl+Shadow-Rice-3 (dBW)']
y4 = P_Rx_data_set['P_Rx_fspl+shadow+ShadowRice-4 (dBW)']
y5 = P_Rx_data_set['P_Rx_fspl+shadow+Jake-5 (dBW)']
y5z = np.array(y5)

# Received Power from Satellite at GS 
# P_rx vs elevation angle
# Option1: fspl and fspl+ShadowedRice and fspl+shadow+ShadowedRice
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(x2[0:165], y1[0:165], label='fspl', linewidth=2, color='blue')
plt.plot(x2[0:165], y3[0:165], label='fspl+ShadowedRice fading')
plt.plot(x2[0:165], y4[0:165], label='fspl+Shadowing+ShadowedRice fading')
plt.xlabel('Elevation Angle (degree)')
plt.ylabel('Received Power [dBW]')
plt.title('Satellite Received Power by GroundStation')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()   
########
# Option2: fspl vs fspl+Shadowed vs fspl+ShadowedRice
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(x2[0:165], y1[0:165], label='fspl', linewidth=2, color='blue')
plt.plot(x2[0:165], y2[0:165], label='fspl+Shadowing', color='m')
plt.plot(x2[0:165], y3[0:165], label='fspl+ShadowedRice fading', color = 'orange')
plt.xlabel('Elevation Angle (degree)')
plt.ylabel('Received Power [dBW]')
plt.title('Satellite Received Power by GroundStation')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show() 
##########
#Option3: fspl vs fspl+shadow+ShadowedRice vs fspl+shadowing+Jake's
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(x2[0:165], y1[0:165], label='fspl', linewidth=2, color='blue')
plt.plot(x2[0:165], y4[0:165], label='fspl+Shadowing+ShadowedRice fading')
#plt.plot(x2[0:165], np.array(y5[0:165]), label='fspl+Shadowing+Jakes fading')
plt.xlabel('Elevation Angle (degree)')
plt.ylabel('Received Power [dBW]')
plt.title('Satellite Received Power by GroundStation')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()  
####
# P_rx vs time steps 
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(y3,label='Satellite Received Power', linewidth=2, color='blue')
plt.xlabel('Time Step [second]')
plt.ylabel('Received Power [dBW]')
plt.title('Satellite Received Power by GroundStation: fspl+ShadowedRice')

plt.show()

########## Shadow + fspl case 2 auto correlation 
## # Plot the auto-correlation
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(y2.astype(float), 100), linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation Value')
plt.title('Satellite P_rx Auto-correlation: fspl+Shadowing')
plt.grid(True)
plt.show()

########## ShadowRice + fspl case 3 auto correlation 
## # Plot the auto-correlation
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(y3.astype(float), 100), linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation Value')
plt.title('Satellite P_rx Auto-correlation: fspl+ShadowedRice')
plt.grid(True)
plt.show()

### fspl+shadow+ShadowedRice case 4
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(y4.astype(float), 100), linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation Value')
plt.title('Satellite P_rx Auto-correlation: fspl+Shadowing+ShadowedRice')
plt.grid(True)
plt.show()

###  fspl+shadowing+Jake's case 5
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(np.array(y5z[436]).astype(float), 100), linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation Value')
plt.title('Satellite P_rx Auto-correlation: fspl+Shadowing+Jakes')
plt.grid(True)
plt.show()
##############################################################################
### Interfernce Power received at GS
# HAPS Interference on GS
xi1 = Interference_HAP_GS_df['distance to GS (km)']
yi1 = np.array(Interference_HAP_GS_df['P_rx (dBW)'])
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(yi1,label='HAPS Interference Power', linewidth=2, color='blue')
plt.xlabel('Time Step [second]')
plt.ylabel('Interference Power [dBW]')
plt.title('HAP interference power on GroundStation')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# HAPS interfernce auro-correlation
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(yi1[60:120].astype(float), 59), linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation Value')
plt.title('Auto-correlation of HAPS Interfernce Power by GS')
plt.grid(True)
plt.show()
############
# UAV3 Interfernce on GS
yi1_uav3 = np.array(Interference_uav3_GS_df['P_rx (dBW)'])
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(yi1_uav3,label='UAV Interference Power', linewidth=2, color='blue')
plt.xlabel('Time Step [second]')
plt.ylabel('Interference Power [dBW]')
plt.title('UAV interference power on GroundStation')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# UAV3 Interference auto-correlation
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(yi1_uav3[240:280].astype(float), 89), linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation Value')
plt.title('Auto-correlation of UAV3 Interfernce Power by GS')
plt.grid(True)
plt.show()
########
# HAP and UAV3 Interefernce Rx power on GS
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(yi1,label='HAPS Interference Power', linewidth=2, color='blue')
plt.plot(yi1_uav3,label='UAV Interference Power', linewidth=2, color='red')
plt.xlabel('Time Step [second]')
plt.ylabel('Interference Power [dBW]')
plt.title('GroundStation Received Interference Power')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()

##############################################################################
###### UAV1
# Received power from HAP by UAV1
y_hap_on_uav1 = HAP_to_uav1_Prx_df['P_rx (dBW)']
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(y_hap_on_uav1[0:100],linewidth=2, color='blue')
plt.xlabel('Time Step')
plt.ylabel('Received Power [dBW]')
plt.title('HAPS Received Power by UAV1')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# Interference Power on uav1 from satellite 
#################
xi2 = Interference_Sat_uav1_df['distance (km)']
yi2 = np.array(Interference_Sat_uav1_df['P_rx (dBW)'])
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(yi2,linewidth=2, color='blue')
plt.xlabel('Time Step')
plt.ylabel('Interference Power [dBW]')
plt.title('Satellite interference power on UAV1')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#######
# P_rx (HAP) vs Intereference power (sateellite)
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(y_hap_on_uav1,linewidth=2, color='blue', label= 'P_rx from HAPS')
plt.plot(yi2,linewidth=2, color='red', label = 'Interference_rx from Satellite')
plt.xlabel('Time Step')
plt.ylabel('Received Power [dBW]')
plt.title('HAPS P_rx and Satellite I_rx on UAV1')
plt.legend(loc='upper left', prop={'size': 7})
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# Plot the auto-correlation
# Received Power from HAP
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(y_hap_on_uav1.astype(float), 10),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of P_rx from HAPS by uav1')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
# Interference from Satellite 
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(yi2.astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of Satellite Interfernce on uav1')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
###############################################################################
##### UAV2 
# Received Power from BaseStation 1
y_bs1_on_uav2 = uav2_Rx_power_fromBS1_df['P_rx (dBW)']
y_i_sat_on_uav2 = Interference_Sat_uav2_df['P_rx (dBW)']
x_I_sat_on_uav2_distnace = Interference_Sat_uav2_df['distance (km)']
x_I_sat_on_uav2_ele_angle = Interference_Sat_uav2_df['Elevation Angle (degree)']
y_i_haps_on_uav2 = Interference_HAP_uav2_df['P_rx (dBW)']
# Received power from BS1 by UAV2
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(y_bs1_on_uav2[0:100],linewidth=2, color='blue')
plt.xlabel('Time Step')
plt.ylabel('Received Power [dBW]')
plt.title('BS1 Received Power by UAV1')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#################
# Interference Power on uav2 from satellite 
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(y_i_sat_on_uav2,linewidth=2, color='blue')
plt.xlabel('Time Step')
plt.ylabel('Interference Power [dBW]')
plt.title('Satellite interference power on UAV2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# same vs elevation angle 
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(x_I_sat_on_uav2_ele_angle[0:165], y_i_sat_on_uav2[0:165],linewidth=2, color='blue')
plt.xlabel('Elevation Angle (Deg)')
plt.ylabel('Interference Power [dBW]')
plt.title('Satellite Interference Power on UAV2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#################
# Interference Power on uav2 from HAPS
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(y_i_haps_on_uav2[0:100],linewidth=2, color='blue')
plt.xlabel('Time Step')
plt.ylabel('Interference Power [dBW]')
plt.title('HAPS interference power on UAV2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
## P_rx from BS1 vs I_rx from Sat and HAPS
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(y_bs1_on_uav2,linewidth=2, color='blue', label = 'P_rx from BaseStation')
plt.plot(y_i_sat_on_uav2,linewidth=2, color='red', label = 'I_rx from Satellite')
plt.plot(y_i_haps_on_uav2,linewidth=2, color='m', label = 'I_rx from HAPS')
plt.xlabel('Time Step')
plt.ylabel('Received Power [dBW]')
plt.title('P_rx from BaseStation vs I_rx from Satellite and HAPS by uav2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
#######################################
# Plot the auto-correlation
# Received Power from BS1
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(y_bs1_on_uav2.astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of P_rx from BaseStation by uav2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
# Interference from Satellite 
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(y_i_sat_on_uav2.astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of Satellite Interfernce on uav2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# Interference from HAPS
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(y_i_haps_on_uav2.astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of HAPS Interfernce on uav2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
##############################################################################
#UEs of BaseStation1
# Plot Heatmap of UEs received power
import seaborn as sns

# Create a heatmap with customized axis labels
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(UEs_BSs1_0, cmap='viridis', ax=ax)

# Customize x-axis labels and ticks

column_labels = [f'UE{i+1}' for i in range(UEs_BSs1_0.shape[1])]
# Set the font size for x-axis labels
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
# Customize y-axis labels and ticks
ax.set_ylabel('Time')
# Set the title
ax.set_title('Heatmap of the UEs around BS1 Received Power')
plt.show()
############
# Plot Histogram of mean of P_rx by UEs over time
pdf_mode = True

plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})

# Use the default color cycle for automatic color assignment
plt.hist(UEs_BSs1_0, bins=10, density=pdf_mode, cumulative=not pdf_mode, label=[f'UE {i+1}' for i in range(UEs_BSs1_0.shape[1])], alpha=0.7)

# Add labels and title
plt.xlabel('P_rx [dB]')
plt.ylabel('PDF')
plt.title('UEs Received Power from BaseStation1')

# Add a legend if both PDF and CDF are plotted
plt.legend(loc='upper left', prop={'size': 7})

# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
##################
# UE1 Receiver Power from BS1
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(UEs_BSs1_0[0:51, 9],linewidth=2, color='blue', label = 'P_rx from BaseStation')
plt.xlabel('Time Step')
plt.ylabel('Received Power [dBW]')
plt.title('P_rx from BaseStation 1 vs I_rx from Satellite and HAPS by uav2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
##################
# auro-correlation plot
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(UEs_BSs1_0[0:51, 9].astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of P_rx from BaseStation1 by UE1')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
##############################
# Plot Heatmap of UEs of BS1 received Interfernce from Satellite and HAPS
# from satellite
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(UEs_of_BS1_Sat_Interferer_0, cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE {i+1}' for i in range(UEs_of_BS1_Sat_Interferer_0.shape[1])]
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
ax.set_ylabel('Time')
ax.set_title('Heatmap of UEs of BS1 Received Interference from Satellite')
plt.show()

# from HAPS
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(UEs_of_BS1_HAPS_Interferer_0, cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE {i+1}' for i in range(UEs_of_BS1_HAPS_Interferer_0.shape[1])]
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
ax.set_ylabel('Time')
ax.set_title('Heatmap of UEs of BS1 Received Interference from HAPS')
plt.show()
###########
# Plot the histogram of received Interfernce from Satellite and HAPS by UEs of BS1
pdf_mode = True
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.hist(UEs_of_BS1_Sat_Interferer_0, bins=10, density=pdf_mode, cumulative=not pdf_mode, label=[f'UE {i+1}' for i in range(UEs_of_BS1_Sat_Interferer_0.shape[1])], alpha=0.7)
plt.hist(UEs_of_BS1_HAPS_Interferer_0, bins=10, density=pdf_mode, cumulative=not pdf_mode, alpha=0.7)
plt.xlabel('P_rx [dB]')
plt.ylabel('PDF')
plt.title('UEs Received Interfernce from Satellite')
plt.legend(loc='upper left', prop={'size': 7})
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
##################
# auro-correlation plot
# Satellite
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(UEs_of_BS1_Sat_Interferer_0[0:300,0].astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of I_rx from Satellite by UE1')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
##################
# auro-correlation plot
# HAPS
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(UEs_of_BS1_HAPS_Interferer_0[0:10,0].astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of I_rx from HAPS by UE1')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
##############################################################################
#UEs of BaseStation2
# Plot Heatmap of UEs received power

# Create a heatmap with customized axis labels
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(UEs_BSs2_0, cmap='viridis', ax=ax)

# Customize x-axis labels and ticks
column_labels = [f'UE{i+1}' for i in range(UEs_BSs2_0.shape[1])]
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
# Customize y-axis labels and ticks
ax.set_ylabel('Time')
# Set the title
ax.set_title('Heatmap of the UEs around BS2 Received Power')
plt.show()
############
# Plot Histogram of mean of P_rx by UEs over time
pdf_mode = True

plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})

# Use the default color cycle for automatic color assignment
plt.hist(UEs_BSs2_0, bins=10, density=pdf_mode, cumulative=not pdf_mode, label=[f'UE {i+1}' for i in range(UEs_BSs2_0.shape[1])], alpha=0.7)

# Add labels and title
plt.xlabel('P_rx [dB]')
plt.ylabel('PDF')
plt.title('UEs Received Power from BaseStation1')

# Add a legend if both PDF and CDF are plotted
plt.legend(loc='upper left', prop={'size': 7})

# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
##################
# UE1 Receiver Power from BS2
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(UEs_BSs2_0[0:51, 9],linewidth=2, color='blue', label = 'P_rx from BaseStation')
plt.xlabel('Time Step')
plt.ylabel('Received Power [dBW]')
plt.title('P_rx from BaseStation 2 by UE10')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
##################
# auro-correlation plot
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(UEs_BSs2_0[0:51, 9].astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of P_rx from BaseStation1 by UE1')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
##############################
# Plot Heatmap of UEs of BS1 received Interfernce from Satellite and HAPS
# from satellite
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(UEs_of_BS2_Sat_Interferer_0, cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE {i+1}' for i in range(UEs_of_BS2_Sat_Interferer_0.shape[1])]
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
ax.set_ylabel('Time')
ax.set_title('Heatmap of UEs of BS1 Received Interference from Satellite')
plt.show()

# from HAPS
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(UEs_of_BS2_HAPS_Interferer_0, cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE {i+1}' for i in range(UEs_of_BS2_HAPS_Interferer_0.shape[1])]
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
ax.set_ylabel('Time')
ax.set_title('Heatmap of UEs of BS1 Received Interference from HAPS')
plt.show()
###########
# Plot the histogram of received Interfernce from Satellite and HAPS by UEs of BS1
pdf_mode = True
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.hist(UEs_of_BS2_Sat_Interferer_0, bins=10, density=pdf_mode, cumulative=not pdf_mode, label=[f'UE {i+1}' for i in range(UEs_of_BS2_Sat_Interferer_0.shape[1])], alpha=0.7)
plt.hist(UEs_of_BS2_HAPS_Interferer_0, bins=10, density=pdf_mode, cumulative=not pdf_mode, alpha=0.7)
plt.xlabel('P_rx [dB]')
plt.ylabel('PDF')
plt.title('UEs of BS2 Received Interfernce from Satellite and HAPS')
plt.legend(loc='upper left', prop={'size': 7})
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
##################
# auro-correlation plot
# Satellite
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(UEs_of_BS2_Sat_Interferer_0[0:300,0].astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of I_rx from Satellite by UE1 of BS2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
##################
# auro-correlation plot
# HAPS
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(UEs_of_BS2_HAPS_Interferer_0[0:10,0].astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of I_rx from HAPS by UE1 of BS2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
##############################################################################
#UEs of BaseStation3
# Plot Heatmap of UEs received power
# Create a heatmap with customized axis labels
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(UEs_BSs3_0, cmap='viridis', ax=ax)

# Customize x-axis labels and ticks
column_labels = [f'UE{i+1}' for i in range(UEs_BSs3_0.shape[1])]
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
# Customize y-axis labels and ticks
ax.set_ylabel('Time')
# Set the title
ax.set_title('Heatmap of the UEs around BS1 Received Power')
plt.show()
############
# Plot Histogram of mean of P_rx by UEs over time
pdf_mode = True

plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})

# Use the default color cycle for automatic color assignment
plt.hist(UEs_BSs3_0, bins=10, density=pdf_mode, cumulative=not pdf_mode, label=[f'UE {i+1}' for i in range(UEs_BSs3_0.shape[1])], alpha=0.7)

# Add labels and title
plt.xlabel('P_rx [dB]')
plt.ylabel('PDF')
plt.title('UEs Received Power from BaseStation1')

# Add a legend if both PDF and CDF are plotted
plt.legend(loc='upper left', prop={'size': 7})

# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
##################
# UE1 Receiver Power from BS1
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(UEs_BSs3_0[0:51, 9],linewidth=2, color='blue', label = 'P_rx from BaseStation')
plt.xlabel('Time Step')
plt.ylabel('Received Power [dBW]')
plt.title('P_rx from BaseStation 3 by UE10')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
##################
# auro-correlation plot
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(UEs_BSs1_0[0:51, 9].astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of P_rx from BaseStation3 by UE10')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
##############################
# Plot Heatmap of UEs of BS1 received Interfernce from Satellite and HAPS
# from satellite
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(UEs_of_BS3_Sat_Interferer_0, cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE {i+1}' for i in range(UEs_of_BS3_Sat_Interferer_0.shape[1])]
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
ax.set_ylabel('Time')
ax.set_title('Heatmap of UEs of BS3 Received Interference from Satellite')
plt.show()
###########
# Plot the histogram of received Interfernce from Satellite and HAPS by UEs of BS1
pdf_mode = True
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.hist(UEs_of_BS3_Sat_Interferer_0, bins=10, density=pdf_mode, cumulative=not pdf_mode, label=[f'UE {i+1}' for i in range(UEs_of_BS3_Sat_Interferer_0.shape[1])], alpha=0.7)
plt.xlabel('P_rx [dB]')
plt.ylabel('PDF')
plt.title('UEs of BS3 Received Interfernce from Satellite')
plt.legend(loc='upper left', prop={'size': 7})
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
##################
# auro-correlation plot
# Satellite
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(UEs_of_BS3_Sat_Interferer_0[0:300,0].astype(float), 100),linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation')
plt.title('Auto-correlation of I_rx from Satellite by UEs of BS3')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', prop={'size': 7})
plt.show()
##################
#%%
##############################################################################
# Received Power and SNR for UEs on the ground from Satellite and HAPS
# data preparation:
    # satellite
Satellite_to_UEs_Prx_aux = np.array(Satellite_to_UEs_Prx_SNR_df['P_rx_total (dBW)']) #auxilary variable
Satellite_to_UEs_Prx_main = np.array([w.reshape(num_base_stations, num_points) for w in Satellite_to_UEs_Prx_aux])
# Cell 1
Satellite_to_BS1_UEs_Prx = Satellite_to_UEs_Prx_main[:,0,:]
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(Satellite_to_BS1_UEs_Prx, cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE {i+1}' for i in range(Satellite_to_BS1_UEs_Prx.shape[1])]
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
ax.set_ylabel('Time')
ax.set_title('Heatmap of P_rx of Cell_1 UEs from Satellite')
plt.show()

# Cell2
Satellite_to_BS2_UEs_Prx = Satellite_to_UEs_Prx_main[:,1,:]
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(Satellite_to_BS2_UEs_Prx, cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE {i+1}' for i in range(Satellite_to_BS2_UEs_Prx.shape[1])]
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
ax.set_ylabel('Time')
ax.set_title('Heatmap of P_rx of Cell_2 UEs from Satellite')
plt.show()

# Cell3
Satellite_to_BS3_UEs_Prx = Satellite_to_UEs_Prx_main[:,2,:]
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(Satellite_to_BS3_UEs_Prx, cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE {i+1}' for i in range(Satellite_to_BS3_UEs_Prx.shape[1])]
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
ax.set_ylabel('Time')
ax.set_title('Heatmap of P_rx of Cell_3 UEs from Satellite')
plt.show()
################ SNR ########################
Satellite_to_UEs_SNR_aux = np.array(Satellite_to_UEs_Prx_SNR_df['SNR (dB)']) #auxilary variable
Satellite_to_UEs_SNR_main = np.array([w.reshape(num_base_stations, num_points) for w in Satellite_to_UEs_SNR_aux])

#Cell 1
Satellite_to_BS1_UEs_SNR = Satellite_to_UEs_SNR_main[:,0,:]

# CDF of SNR and Rate
# CDF of SNR:
sorted_Satellite_to_BS1_UEs_SNR = np.sort(Satellite_to_BS1_UEs_SNR, axis  = 0)
Satellite_to_BS1_UEs_SNR_cdf = np.arange(1, len(sorted_Satellite_to_BS1_UEs_SNR) + 1) / len(sorted_Satellite_to_BS1_UEs_SNR)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_Satellite_to_BS1_UEs_SNR[:, col], Satellite_to_BS1_UEs_SNR_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
plt.title('UEs of Cell 1 SNR CDF from Satellite', fontsize=12)
plt.xlabel('SNR [dB]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()

# CDF of rate:
Satellite_to_BS1_UEs_SNR_linear = 10**(Satellite_to_BS1_UEs_SNR/10)
Rate_Satellite_to_BS1_UEs = ((max_Bandwidth_per_beam)*np.log2(1 + Satellite_to_BS1_UEs_SNR_linear)).astype(float)
sorted_Rate_Satellite_to_BS1_UEs = np.sort(Rate_Satellite_to_BS1_UEs, axis = 0)
# Calculate the cumulative distribution
Rate_Satellite_to_BS1_UEs_cdf = np.arange(1, len(sorted_Rate_Satellite_to_BS1_UEs) + 1) / len(sorted_Rate_Satellite_to_BS1_UEs)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_Rate_Satellite_to_BS1_UEs[:, col], Rate_Satellite_to_BS1_UEs_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
plt.title('UEs of Cell_1 Maximum Acheivable Data Rate CDF from Satellite', fontsize=12)
plt.xlabel('Data Rate [Mbit/sec]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()

# Cell 2
Satellite_to_BS2_UEs_SNR = Satellite_to_UEs_SNR_main[:,1,:]
Satellite_to_BS2_UEs_SNR_linear = 10**(Satellite_to_BS2_UEs_SNR/10)
Rate_Satellite_to_BS2_UEs = ((max_Bandwidth_per_beam/num_points)*np.log2(1 + Satellite_to_BS2_UEs_SNR_linear)).astype(float)
sorted_Rate_Satellite_to_BS2_UEs = np.sort(Rate_Satellite_to_BS2_UEs, axis = 0)
# Calculate the cumulative distribution
Rate_Satellite_to_BS2_UEs_cdf = np.arange(1, len(sorted_Rate_Satellite_to_BS2_UEs) + 1) / len(sorted_Rate_Satellite_to_BS2_UEs)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_Rate_Satellite_to_BS2_UEs[:, col], Rate_Satellite_to_BS2_UEs_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
plt.title('UEs of Cell_2 Average Acheivable Data Rate CDF from Satellite', fontsize=12)
plt.xlabel('Data Rate [Mbit/sec]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()

# Cell_3
Satellite_to_BS3_UEs_SNR = Satellite_to_UEs_SNR_main[:,2,:]
# CDF of SNR:
sorted_Satellite_to_BS3_UEs_SNR = np.sort(Satellite_to_BS3_UEs_SNR, axis  = 0)
Satellite_to_BS3_UEs_SNR_cdf = np.arange(1, len(sorted_Satellite_to_BS3_UEs_SNR) + 1) / len(sorted_Satellite_to_BS3_UEs_SNR)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_Satellite_to_BS3_UEs_SNR[:, col], Satellite_to_BS3_UEs_SNR_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
plt.title('UEs on the ground SNR from Satellite', fontsize=12)
plt.xlabel('SNR [dB]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()

# CDF of Rate
Satellite_to_BS3_UEs_SNR_linear = 10**(Satellite_to_BS3_UEs_SNR/10)
Rate_Satellite_to_BS3_UEs = ((max_Bandwidth_per_beam/3)*np.log2(1 + Satellite_to_BS3_UEs_SNR_linear)).astype(float)
sorted_Rate_Satellite_to_BS3_UEs = np.sort(Rate_Satellite_to_BS3_UEs, axis = 0)
# Calculate the cumulative distribution
Rate_Satellite_to_BS3_UEs_cdf = np.arange(1, len(sorted_Rate_Satellite_to_BS3_UEs) + 1) / len(sorted_Rate_Satellite_to_BS3_UEs)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_Rate_Satellite_to_BS3_UEs[:, col], Rate_Satellite_to_BS3_UEs_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
plt.title('UEs of Cell_3 Maximum Acheivable Data Rate CDF from Satellite per Cell', fontsize=12)
plt.xlabel('Data Rate [Mbit/sec]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()

    # HAPS
HAPS_to_UEs_Prx_aux = np.array(HAPS_to_UEs_Prx_SNR_df['P_rx_total (dBW)']) #auxilary variable
HAPS_to_UEs_Prx_main = np.array([w.reshape(num_base_stations, num_points) for w in HAPS_to_UEs_Prx_aux])

HAPS_to_BS1_UEs_Prx = HAPS_to_UEs_Prx_main[:,0,:]
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(HAPS_to_BS1_UEs_Prx, cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE {i+1}' for i in range(HAPS_to_BS1_UEs_Prx.shape[1])]
ax.set_xticklabels(column_labels, fontsize=8)
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
ax.set_ylabel('Time')
ax.set_title('Heatmap of P_rx of BS1 UEs from Satellite')
plt.show()

HAPS_to_BS2_UEs_Prx = HAPS_to_UEs_Prx_main[:,1,:]

HAPS_to_BS3_UEs_Prx = HAPS_to_UEs_Prx_main[:,2,:]
# SNR
HAPS_to_UEs_SNR_aux = np.array(HAPS_to_UEs_Prx_SNR_df['SNR (dB)']) #auxilary variable
HAPS_to_UEs_SNR_main = np.array([w.reshape(num_base_stations, num_points) for w in HAPS_to_UEs_SNR_aux])

#Cell 1
HAPS_to_BS1_UEs_SNR = HAPS_to_UEs_SNR_main[:,0,:]
# CDF of SNR and Rate
# CDF of SNR:
sorted_HAPS_to_BS1_UEs_SNR = np.sort(HAPS_to_BS1_UEs_SNR, axis  = 0)
HAPS_to_BS1_UEs_SNR_cdf = np.arange(1, len(sorted_HAPS_to_BS1_UEs_SNR) + 1) / len(sorted_HAPS_to_BS1_UEs_SNR)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_HAPS_to_BS1_UEs_SNR[:, col], HAPS_to_BS1_UEs_SNR_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
plt.title('UEs of Cell 1 SNR CDF from HAPS', fontsize=12)
plt.xlabel('SNR [dB]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()

# CDF of rate:
HAPS_to_BS1_UEs_SNR_linear = 10**(HAPS_to_BS1_UEs_SNR/10)
Rate_HAPS_to_BS1_UEs = ((max_Bandwidth_per_beam)*np.log2(1 + HAPS_to_BS1_UEs_SNR_linear)).astype(float)
sorted_Rate_HAPS_to_BS1_UEs = np.sort(Rate_HAPS_to_BS1_UEs, axis = 0)
# Calculate the cumulative distribution
Rate_HAPS_to_BS1_UEs_cdf = np.arange(1, len(sorted_Rate_HAPS_to_BS1_UEs) + 1) / len(sorted_Rate_HAPS_to_BS1_UEs)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_Rate_HAPS_to_BS1_UEs[:, col], Rate_HAPS_to_BS1_UEs_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
plt.title('UEs of Cell_1 Maximum Acheivable Data Rate CDF from HAPS', fontsize=12)
plt.xlabel('Data Rate [Mbit/sec]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()
# Cell_2
HAPS_to_BS2_UEs_SNR = HAPS_to_UEs_SNR_main[:,1,:]
HAPS_to_BS3_UEs_SNR = HAPS_to_UEs_SNR_main[:,2,:]

# %% SINR and Data Rate Histogram, pdf and cdf plot

# 1- Ground Station SIR and Data Rate
x_sir_hap = 10*np.log10(np.array(Ground_Station_SIR_Rate['SIR_HAP']))
x_r_hap = np.array(Ground_Station_SIR_Rate['Data Rate - Int-is-HAP (Mbit/sec)'])
x_sir_uav3 = 10*np.log10(np.array(Ground_Station_SIR_Rate['SIR_uav3']))
#x_Outage_sir_uav3 = 10 * np.log10(Ground_Station_SIR_Rate['SIR_uav3'][Ground_Station_SIR_Rate['SIR_uav3'] <= 1].values)
x_r_uav3 = np.array(Ground_Station_SIR_Rate['Data Rate - Int-is-uav3 (Mbit/sec)'])
x_sir_bs = 10*np.log10(np.array(Ground_Station_SIR_Rate['SIR_BS']))
x_r_bs = np.array(Ground_Station_SIR_Rate['Data Rate - Int-is-BS (Mbit/sec)'])
x_sir_total = 10*np.log10(np.array(Ground_Station_SIR_Rate['SIR_total']))
x_r_total = np.array(Ground_Station_SIR_Rate['Data Rate - Int-is-Total (Mbit/sec)'])
# Plot the histogram for SINR
pdf_mode = False
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.hist(x_sir_uav3, bins=30, edgecolor='black', color='#1f77b4',density=pdf_mode, cumulative=not pdf_mode, label='SINR under UAV Interference', alpha=0.7)
plt.hist(x_sir_hap, bins=30, edgecolor='black', color='red',density=pdf_mode, cumulative=not pdf_mode, label='SINR under HAPS Interference', alpha=0.7)
plt.hist(x_sir_bs, bins=30, edgecolor='black', color='orange',density=pdf_mode, cumulative=not pdf_mode, label='SINR under BS Interference', alpha=0.7)
plt.hist(x_sir_total, bins=30, edgecolor='black', color='m',density=pdf_mode, cumulative=not pdf_mode, label='Outage SINR', alpha=0.7)
# Add labels and title
plt.xlabel('SINR [dB]')
plt.ylabel('CDF')
plt.title('GroundStation Outage Probability CDF')
# Add a legend if both PDF and CDF are plotted
plt.legend(prop={'size': 7})
# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot CDF of data Rate
# Sort the data
sorted_data_hap = np.sort(x_r_hap)
sorted_data_uav3 = np.sort(x_r_uav3)
#sorted_data_bs = np.sort(x_r_bs[x_r_bs<1000])
sorted_data_bs = np.sort(x_r_bs)
sorted_data_total = np.sort(x_r_total)
# Calculate the cumulative distribution
gs_rate_hap_cdf = np.arange(1, len(sorted_data_hap) + 1) / len(sorted_data_hap)
gs_rate_uav_cdf = np.arange(1, len(sorted_data_uav3) + 1) / len(sorted_data_uav3)
gs_rate_bs_cdf = np.arange(1, len(sorted_data_bs) + 1) / len(sorted_data_bs)
gs_rate_total_cdf = np.arange(1, len(sorted_data_total) + 1) / len(sorted_data_total)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(sorted_data_hap, gs_rate_hap_cdf, label='HAPS to Ground', linewidth=2, color='blue')
plt.plot(sorted_data_uav3, gs_rate_uav_cdf, label='UAV to UAV', linewidth=2, color='red')
plt.plot(sorted_data_bs, gs_rate_bs_cdf, label='gNB to UE', linewidth=2, color='k')
plt.plot(sorted_data_total, gs_rate_total_cdf, label='Total Interference', linewidth=2, color='g')
plt.title('LEO to Ground-Station Communication', fontsize=12)
plt.xlabel('Data Rate [Mbit/sec]', fontsize=14)
plt.ylabel('ECDF', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()

##############################################################################
# 2 - UAV1:
x_sir_satellite = UAV1_SIR_Rate['SIR_satellite']
x_r_satellite = UAV1_SIR_Rate['Data Rate - Int-is-satellite (Mbit/sec)']
#### Plot SINR
# Plot the histogram for SINR
pdf_mode = False
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.hist(10*np.log10(x_sir_satellite), bins=50, edgecolor='black', color='#1f77b4',density=pdf_mode, cumulative=not pdf_mode, label='SINR under Satellite Interference', alpha=0.7)
# Add labels and title
plt.xlabel('SINR [dB]')
plt.ylabel('CDF')
plt.title('UAV1 Outage Probability under Satellite Interference')
# Add a legend if both PDF and CDF are plotted
plt.legend(loc='upper left', prop={'size': 7})
# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# CDF of data Rate
sorted_data_satellite_on_uav1 = np.sort(x_r_satellite)
# Calculate the cumulative distribution
uav1_rate_sat_cdf = np.arange(1, len(sorted_data_satellite_on_uav1) + 1) / len(sorted_data_satellite_on_uav1)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(sorted_data_satellite_on_uav1, uav1_rate_sat_cdf, label='Satellite Interference', linewidth=2, color='blue')
plt.title('UAV1 Acheivable Data Rate CDF in Presence of Interfernce from Satellite', fontsize=12)
plt.xlabel('Data Rate [Mbit/sec]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()
##############################################################################
# 3 - UAV2
x_sir_sat_uav2 = UAV2_SIR_Rate['SIR_satellite']
x_r_sat_uav2 = UAV2_SIR_Rate['Data Rate - Int-is-satellite (Mbit/sec)']
x_sir_haps_uav2 = UAV2_SIR_Rate['SIR_HAP']
x_r_haps_uav2 = UAV2_SIR_Rate['Data Rate - Int-is-HAP (Mbit/sec)']
x_sir_total_uav2 = UAV2_SIR_Rate['SIR_Total'][UAV2_SIR_Rate['SIR_Total']<= 1].values
x_r_total_uav2 = UAV2_SIR_Rate['Data Rate - Int-is-total (Mbit/sec)']
#### Plot SINR
# Plot the histogram for SINR
pdf_mode = False
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.hist(10*np.log10(x_sir_sat_uav2), bins=50, edgecolor='black', color='#1f77b4',density=pdf_mode, cumulative=not pdf_mode, label='SINR under Satellite Interference', alpha=0.7)
plt.hist(10*np.log10(x_sir_haps_uav2), bins=50, edgecolor='black', color='red',density=pdf_mode, cumulative=not pdf_mode, label='SINR under HAPS Interference', alpha=0.7)
plt.hist(10*np.log10(x_sir_total_uav2), bins=50, edgecolor='black', color='m',density=pdf_mode, cumulative=not pdf_mode, label='SINR under Satellite and HAPS Interferences', alpha=0.7)

# Add labels and title
plt.xlabel('SINR [dB]')
plt.ylabel('CDF')
plt.title('UAV2 Outage Probability pdf under Satellite and HAPS Interference')
# Add a legend if both PDF and CDF are plotted
plt.legend(loc='upper left', prop={'size': 7})
# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# Plot CDF of data Rate
# Sort the data
sorted_data_haps_on_uav2 = np.sort(x_r_haps_uav2)
sorted_data_sat_on_uav2 = np.sort(x_r_sat_uav2)
sorted_data_total_uav2 = np.sort(x_r_total_uav2)
# Calculate the cumulative distribution
uav2_rate_haps_cdf = np.arange(1, len(sorted_data_haps_on_uav2) + 1) / len(sorted_data_haps_on_uav2)
uav2_rate_sat_cdf = np.arange(1, len(sorted_data_sat_on_uav2) + 1) / len(sorted_data_sat_on_uav2)
uav2_rate_total_cdf = np.arange(1, len(sorted_data_total_uav2) + 1) / len(sorted_data_total_uav2)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(sorted_data_sat_on_uav2, uav2_rate_sat_cdf, label='Satellite Interference', linewidth=2, color='blue')
plt.plot(sorted_data_haps_on_uav2, uav2_rate_haps_cdf, label='HAPS Interference', linewidth=2, color='red')
plt.plot(sorted_data_total_uav2, uav2_rate_total_cdf, label='Total Interference', linewidth=2, color='m')
plt.title('UAV2 Acheivable Data Rate CDF in Presence of Interfernce from HAPS and Satellite', fontsize=12)
plt.xlabel('Data Rate [Mbit/sec]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()
##############################################################################
# 4 - UEs of BS1
# Create a heatmap with customized axis labels
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(10*np.log10(SIR_UEs_BS1_Sate_and_HAP_0), cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE{i+1}' for i in range(SIR_UEs_BS1_Sate_and_HAP_0.shape[1])]
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
# Customize y-axis labels and ticks
ax.set_ylabel('Time')
# Set the title
ax.set_title('SIR of UEs of BS1 [dB]')
plt.show()
############
# Plot Histogram of SIR of UEs over time
pdf_mode = False
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
# Use the default color cycle for automatic color assignment
plt.hist(10*np.log10(SIR_UEs_BS1_Sate_and_HAP_0), bins=10, density=pdf_mode, cumulative=not pdf_mode, label=[f'UE {i+1}' for i in range(SIR_UEs_BS1_Sate_and_HAP_0.shape[1])], alpha=0.7)
# Add labels and title
plt.xlabel('SIR [dB]')
plt.ylabel('PDF')
plt.title('UEs of BS1 SIR CDF')
# Add a legend if both PDF and CDF are plotted
plt.legend(loc='upper left', prop={'size': 7})
# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#############
# plot the Data Rate
# CDF of data Rate
Rate_UEs_BS1 = ((max_Bandwidth_per_beam/num_points)*np.log2(1 + SIR_UEs_BS1_Sate_and_HAP_0)).astype(float)
sorted_data_Rate_UEs_BS1 = np.sort(Rate_UEs_BS1, axis = 0)
# Calculate the cumulative distribution
Rate_UEs_BS1_cdf = np.arange(1, len(sorted_data_Rate_UEs_BS1) + 1) / len(sorted_data_Rate_UEs_BS1)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_data_Rate_UEs_BS1[:, col], Rate_UEs_BS1_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
plt.title('UEs of BS1 Acheivable Data Rate CDF in Presence of I from Satellite and HAPS', fontsize=12)
plt.xlabel('Data Rate [Mbit/sec]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()
##############################################################################
# 5 - UEs of BS2
# Create a heatmap with customized axis labels
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(10*np.log10(SIR_UEs_BS2_Sate_and_HAP_0), cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE{i+1}' for i in range(SIR_UEs_BS2_Sate_and_HAP_0.shape[1])]
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
# Customize y-axis labels and ticks
ax.set_ylabel('Time')
# Set the title
ax.set_title('SIR of UEs of BS2 [dB]')
plt.show()
############
# Plot Histogram of SIR of UEs over time
pdf_mode = False
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
# Use the default color cycle for automatic color assignment
plt.hist(10*np.log10(SIR_UEs_BS2_Sate_and_HAP_0), bins=10, density=pdf_mode, cumulative=not pdf_mode, label=[f'UE {i+1}' for i in range(SIR_UEs_BS2_Sate_and_HAP_0.shape[1])], alpha=0.7)
# Add labels and title
plt.xlabel('SIR [dB]')
plt.ylabel('PDF')
plt.title('UEs of BS2 SIR CDF')
# Add a legend if both PDF and CDF are plotted
plt.legend(loc='upper left', prop={'size': 7})
# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#############
# plot the Data Rate
# CDF of data Rate
Rate_UEs_BS2 = ((max_Bandwidth_per_beam/num_points)*np.log2(1 + SIR_UEs_BS2_Sate_and_HAP_0)).astype(float)
sorted_data_Rate_UEs_BS2 = np.sort(Rate_UEs_BS2, axis = 0)
# Calculate the cumulative distribution
Rate_UEs_BS2_cdf = np.arange(1, len(sorted_data_Rate_UEs_BS2) + 1) / len(sorted_data_Rate_UEs_BS2)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_data_Rate_UEs_BS2[:, col], Rate_UEs_BS2_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
plt.title('UEs of BS2 Acheivable Data Rate CDF in Presence of I from Satellite and HAPS', fontsize=12)
plt.xlabel('Data Rate [Mbit/sec]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()
##############################################################################
# UEs of BS3
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(10*np.log10(SIR_UEs_BS3_Sate_0), cmap='viridis', ax=ax)
# Customize x-axis labels and ticks
column_labels = [f'UE{i+1}' for i in range(SIR_UEs_BS3_Sate_0.shape[1])]
ax.set_xticks(np.arange(0.5, len(column_labels), 1))
ax.set_xticklabels(column_labels)
ax.set_xlabel('UEs')
# Customize y-axis labels and ticks
ax.set_ylabel('Time')
# Set the title
ax.set_title('SIR of UEs of BS3 [dB]')
plt.show()
############
# Plot CDF of SINR of UEs getting service from BaseStation and Interfered by Satellite
# CDF of SINR
sorted_SIR_UEs_BS3_Sate_0 = np.sort(10*np.log10(SIR_UEs_BS3_Sate_0), axis = 0)
# Calculate the cumulative distribution
SIR_UEs_BS3_Sate_0_cdf = np.arange(1, len(sorted_SIR_UEs_BS3_Sate_0) + 1) / len(sorted_SIR_UEs_BS3_Sate_0)
# Plot the CDF
plt.figure(figsize=(3.36, 2.3), dpi=300)
plt.rcParams.update({
    'font.size': 5,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_SIR_UEs_BS3_Sate_0[:, col], SIR_UEs_BS3_Sate_0_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
#plt.title('UEs SINR in presence of Satellite Interference', fontsize=12)
plt.xlabel('SINR [dB]', fontsize=5)
plt.ylabel('ECDF', fontsize=5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 5})
# Save the figure in both .pgf and high-quality .png formats
pgf_path = os.path.join(save_directory, 'figure2.pgf')
png_path = os.path.join(save_directory, 'figure2.png')
plt.savefig(pgf_path)
plt.savefig(png_path, dpi=300)
plt.show()    
#############
# plot the Data Rate
# CDF of data Rate
Rate_UEs_BS3 = ((max_Bandwidth_per_beam)*np.log2(1 + SIR_UEs_BS3_Sate_0)).astype(float)
sorted_data_Rate_UEs_BS3 = np.sort(Rate_UEs_BS3, axis = 0)
# Calculate the cumulative distribution
Rate_UEs_BS3_cdf = np.arange(1, len(sorted_data_Rate_UEs_BS3) + 1) / len(sorted_data_Rate_UEs_BS3)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_data_Rate_UEs_BS3[:, col], Rate_UEs_BS3_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
plt.title('UEs of BS3 Maximum Acheivable Data Rate CDF in Presence of Satellite Interference', fontsize=12)
plt.xlabel('Data Rate [Mbit/sec]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()
#%%
############################################ Now if we consider satellite as main service provider and BaseStation as Interferer: 
# Plot CDF of SINR of UEs getting service from Satellite and Interfered by BaseStation
# CDF of SINR
SIR_Satellite_UEs_BS3 = np.divide(1,SIR_UEs_BS3_Sate_0)
sorted_SIR_Satellite_UEs_BS3_0 = np.sort(10*np.log10(SIR_Satellite_UEs_BS3), axis = 0)
# Calculate the cumulative distribution
SIR_Satellite_UEs_BS3_0_cdf = np.arange(1, len(sorted_SIR_Satellite_UEs_BS3_0) + 1) / len(sorted_SIR_Satellite_UEs_BS3_0)

# Plot the CDF
plt.figure(figsize=(3.36, 2.3), dpi=300)
plt.rcParams.update({
    'font.size': 5,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_SIR_Satellite_UEs_BS3_0[:, col], SIR_Satellite_UEs_BS3_0_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
#plt.title('UEs SINR in presence of BaseStation Interference', fontsize=12)
plt.xlabel('SINR [dB]', fontsize=5)
plt.ylabel('ECDF', fontsize=5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 5})
# Save the figure in both .pgf and high-quality .png formats
pgf_path = os.path.join(save_directory, 'figure.pgf')
png_path = os.path.join(save_directory, 'figure.png')
plt.savefig(pgf_path)
plt.savefig(png_path, dpi=300)
plt.show()    


    
# plot the Data Rate
# CDF of data Rate
Rate_Satellite_UEs_BS3 = ((max_Bandwidth_per_beam)*np.log2(1 + np.divide(1,SIR_UEs_BS3_Sate_0))).astype(float)
sorted_data_Rate_Satellite_UEs_BS3 = np.sort(Rate_Satellite_UEs_BS3, axis = 0)
# Calculate the cumulative distribution
Rate_Satellite_UEs_BS3_cdf = np.arange(1, len(sorted_data_Rate_Satellite_UEs_BS3) + 1) / len(sorted_data_Rate_Satellite_UEs_BS3)
# Plot the CDF
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))
for col in range(num_points):
    plt.plot(sorted_data_Rate_Satellite_UEs_BS3[:, col], Rate_Satellite_UEs_BS3_cdf, label=f'UE {col+1}', linewidth=2, color=colors_list[col])
plt.title('UEs of Cell_3 Maximum Acheivable Data Rate CDF in Presence of BaseStation Interference', fontsize=12)
plt.xlabel('Data Rate [Mbit/sec]', fontsize=14)
plt.ylabel('Cumulative Probability', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(prop={'size': 10})
plt.show()   

#%% Calculate Percentage of Outage SINR for every layer:
# 1- Ground Users --> 1-1. Main: BS - Interf: Sat, HAPS ; 1-2. Main: Sat - Interf: BS, HAPS ; 1-3. Main: HAPS - Interf: BS, Sat ; 1-4. Main: Sat to GS - Interf: HAPS, Drone
# 2- Air Users --> 2-1. Mian: BS - interf: Sat, HAPS ; 2-2. Main: Sat - Interf: BS, HAPS ; 2-3. Main: HAPS - Interf: Sat
"""*******************************************************"""
# We start from 1-4:
x_Outage_sir_hap = 10 * np.log10(Ground_Station_SIR_Rate['SIR_HAP'][Ground_Station_SIR_Rate['SIR_HAP'] < 1].values)
sinr_outage_hap_percent = (len(x_Outage_sir_hap)/len(Ground_Station_SIR_Rate['SIR_HAP']))*100

x_Outage_sir_uav3 = 10 * np.log10(Ground_Station_SIR_Rate['SIR_uav3'][Ground_Station_SIR_Rate['SIR_uav3'] < 1].values)
sinr_outage_uav3_percent = (len(x_Outage_sir_uav3)/len(Ground_Station_SIR_Rate['SIR_uav3']))*100

x_Outage_sir_bs = 10 * np.log10(Ground_Station_SIR_Rate['SIR_BS'][Ground_Station_SIR_Rate['SIR_BS'] < 1].values)
sinr_outage_bs_percent = (len(x_Outage_sir_bs)/len(Ground_Station_SIR_Rate['SIR_BS']))*100

x_Outage_sir_total = 10 * np.log10(Ground_Station_SIR_Rate['SIR_total'][Ground_Station_SIR_Rate['SIR_total'] < 1].values)
sinr_outage_total_percent = (len(x_Outage_sir_total)/len(Ground_Station_SIR_Rate['SIR_total']))*100

# 1-1:
# only Satellites
x_Outage_sir_bs1_ues_from_sat = 10 * np.log10(SINR_UEs_BS1_Sat[SINR_UEs_BS1_Sat < 1])
sinr_outage_UEs_bs1_from_sat_percent = (len(x_Outage_sir_bs1_ues_from_sat)/len(SINR_UEs_BS1_Sat)/num_points)*100

x_Outage_sir_bs2_ues_from_sat = 10 * np.log10(SINR_UEs_BS2_Sat[SINR_UEs_BS2_Sat < 1])
sinr_outage_UEs_bs2_from_sat_percent = (len(x_Outage_sir_bs2_ues_from_sat)/len(SINR_UEs_BS2_Sat)/num_points)*100

x_Outage_sir_bs3_ues_from_sat = 10 * np.log10(SINR_UEs_BS3_Sat[SINR_UEs_BS3_Sat < 1])
sinr_outage_UEs_bs3_from_sat_percent = (len(x_Outage_sir_bs3_ues_from_sat)/len(SINR_UEs_BS3_Sat)/num_points)*100

# only HAPS: BS3 Users are not under HAPS coverage
x_Outage_sir_bs1_ues_from_hap = 10 * np.log10(SINR_UEs_BS1_HAPS[SINR_UEs_BS1_HAPS < 1])
sinr_outage_UEs_bs1_from_hap_percent = (len(x_Outage_sir_bs1_ues_from_hap)/len(SINR_UEs_BS1_HAPS)/num_points)*100

x_Outage_sir_bs2_ues_from_hap = 10 * np.log10(SINR_UEs_BS2_HAPS[SINR_UEs_BS2_HAPS < 1])
sinr_outage_UEs_bs2_from_hap_percent = (len(x_Outage_sir_bs2_ues_from_hap)/len(SINR_UEs_BS2_HAPS)/num_points)*100


# 1-2:
    # only BS
SINR_UEs_Sat_bs1 = np.divide(1,SINR_UEs_BS1_Sat)
x_Outage_sir_sat_ues_from_bs1 = 10 * np.log10(SINR_UEs_Sat_bs1[SINR_UEs_Sat_bs1 < 1])
sinr_outage_UEs_sat_from_bs1_percent = (len(x_Outage_sir_sat_ues_from_bs1)/len(SINR_UEs_Sat_bs1)/num_points)*100

SINR_UEs_Sat_bs2 = np.divide(1,SINR_UEs_BS2_Sat)
x_Outage_sir_sat_ues_from_bs2 = 10 * np.log10(SINR_UEs_Sat_bs2[SINR_UEs_Sat_bs2 < 1])
sinr_outage_UEs_sat_from_bs2_percent = (len(x_Outage_sir_sat_ues_from_bs2)/len(SINR_UEs_Sat_bs2)/num_points)*100

SINR_UEs_Sat_bs3 = np.divide(1,SINR_UEs_BS3_Sat)
x_Outage_sir_sat_ues_from_bs3 = 10 * np.log10(SINR_UEs_Sat_bs3[SINR_UEs_Sat_bs3 < 1])
sinr_outage_UEs_sat_from_bs3_percent = (len(x_Outage_sir_sat_ues_from_bs3)/len(SINR_UEs_Sat_bs3)/num_points)*100

    # only HAPS
SINR_UEs1_Sat_HAPS = np.divide(SINR_UEs_Sat_bs1, np.divide(1,SINR_UEs_BS1_HAPS))
x_Outage_sir_sat_ues1_from_haps = 10 * np.log10(SINR_UEs1_Sat_HAPS[SINR_UEs1_Sat_HAPS < 1])
sinr_outage_UEs1_sat_from_haps_percent = (len(x_Outage_sir_sat_ues1_from_haps)/len(SINR_UEs1_Sat_HAPS)/num_points)*100

SINR_UEs2_Sat_HAPS = np.divide(SINR_UEs_Sat_bs2, np.divide(1,SINR_UEs_BS2_HAPS))
x_Outage_sir_sat_ues2_from_haps = 10 * np.log10(SINR_UEs2_Sat_HAPS[SINR_UEs2_Sat_HAPS < 1])
sinr_outage_UEs2_sat_from_haps_percent = (len(x_Outage_sir_sat_ues2_from_haps)/len(SINR_UEs2_Sat_HAPS)/num_points)*100

SINR_UEs3_Sat_HAPS = np.divide(SINR_UEs_Sat_bs3, np.divide(1,SINR_UEs_BS3_HAPS))
x_Outage_sir_sat_ues3_from_haps = 10 * np.log10(SINR_UEs3_Sat_HAPS[SINR_UEs3_Sat_HAPS < 1])
sinr_outage_UEs3_sat_from_haps_percent = (len(x_Outage_sir_sat_ues3_from_haps)/len(SINR_UEs3_Sat_HAPS)/num_points)*100

# 1-3:
    # only BS
SINR_UEs_Haps_bs1 = np.divide(1,SINR_UEs_BS1_HAPS)
x_Outage_sir_haps_ues_from_bs1 = 10 * np.log10(SINR_UEs_Haps_bs1[SINR_UEs_Haps_bs1 < 1])
sinr_outage_UEs_haps_from_bs1_percent = (len(x_Outage_sir_haps_ues_from_bs1)/len(SINR_UEs_Haps_bs1)/num_points)*100

SINR_UEs_Haps_bs2 = np.divide(1,SINR_UEs_BS2_HAPS)
x_Outage_sir_haps_ues_from_bs2 = 10 * np.log10(SINR_UEs_Haps_bs2[SINR_UEs_Haps_bs2 < 1])
sinr_outage_UEs_haps_from_bs2_percent = (len(x_Outage_sir_haps_ues_from_bs2)/len(SINR_UEs_Haps_bs2)/num_points)*100

SINR_UEs_Haps_bs3 = np.divide(1,SINR_UEs_BS3_HAPS)
x_Outage_sir_haps_ues_from_bs3 = 10 * np.log10(SINR_UEs_Haps_bs3[SINR_UEs_Haps_bs3 < 1])
sinr_outage_UEs_haps_from_bs3_percent = (len(x_Outage_sir_haps_ues_from_bs3)/len(SINR_UEs_Haps_bs3)/num_points)*100

    # only Sat
SINR_UEs1_Haps_sat = np.divide(SINR_UEs_Haps_bs1, SINR_UEs_Sat_bs1)
x_Outage_sir_haps_ues1_from_sat = 10 * np.log10(SINR_UEs1_Haps_sat[SINR_UEs1_Haps_sat < 1])
sinr_outage_UEs1_haps_from_sat_percent = (len(x_Outage_sir_haps_ues1_from_sat)/len(SINR_UEs1_Haps_sat)/num_points)*100

SINR_UEs2_Haps_sat = np.divide(SINR_UEs_Haps_bs2, SINR_UEs_Sat_bs2)
x_Outage_sir_haps_ues2_from_sat = 10 * np.log10(SINR_UEs2_Haps_sat[SINR_UEs2_Haps_sat < 1])
sinr_outage_UEs2_haps_from_sat_percent = (len(x_Outage_sir_haps_ues2_from_sat)/len(SINR_UEs2_Haps_sat)/num_points)*100

SINR_UEs3_Haps_sat =  np.divide(SINR_UEs_Haps_bs3, SINR_UEs_Sat_bs3)
x_Outage_sir_haps_ues3_from_sat = 10 * np.log10(SINR_UEs3_Haps_sat[SINR_UEs3_Haps_sat < 1])
sinr_outage_UEs3_haps_from_sat_percent = (len(x_Outage_sir_haps_ues3_from_sat)/len(SINR_UEs3_Haps_sat)/num_points)*100

# 2-1: for now lets only see for the uav2 which is moving in a circluar path
    # only Sat
x_sir_sat_uav2 = np.array(x_sir_sat_uav2)
x_Outage_sir_uav2_bs1_from_sat = 10 * np.log10(x_sir_sat_uav2[x_sir_sat_uav2 < 1])
sinr_outage_uav2_bs1_from_sat_percent = (len(x_Outage_sir_uav2_bs1_from_sat)/len(x_sir_sat_uav2))*100

    # only HAPS
x_sir_haps_uav2 = np.array(x_sir_haps_uav2)
x_Outage_sir_uav2_bs1_from_haps = 10 * np.log10(x_sir_haps_uav2[x_sir_haps_uav2 < 1])
sinr_outage_uav2_bs1_from_haps_percent = (len(x_Outage_sir_uav2_bs1_from_haps)/len(x_sir_haps_uav2))*100

# 2-2:
    # only BS
SINR_uav2_Sat_bs1 = np.divide(1,x_sir_sat_uav2)
x_Outage_sir_uav2_sat_from_bs1 = 10 * np.log10(SINR_uav2_Sat_bs1[SINR_uav2_Sat_bs1 < 1])
sinr_outage_uav2_sat_from_bs1_percent = (len(x_Outage_sir_uav2_sat_from_bs1)/len(SINR_uav2_Sat_bs1))*100

    # only HAPS
SINR_uav2_Sat_HAPS = np.divide(SINR_uav2_Sat_bs1,np.divide(1,x_sir_haps_uav2))
x_Outage_sir_uav2_sat_from_haps = 10 * np.log10(SINR_uav2_Sat_HAPS[SINR_uav2_Sat_HAPS < 1])
sinr_outage_uav2_sat_from_haps_percent = (len(x_Outage_sir_uav2_sat_from_haps)/len(SINR_uav2_Sat_HAPS))*100

# 2-3:
    # only BS
SINR_uav2_HAPS_bs1 = np.divide(1,x_sir_haps_uav2)
x_Outage_sir_uav2_haps_from_bs1 = 10 * np.log10(SINR_uav2_HAPS_bs1[SINR_uav2_HAPS_bs1 < 1])
sinr_outage_uav2_haps_from_bs1_percent = (len(x_Outage_sir_uav2_haps_from_bs1)/len(SINR_uav2_HAPS_bs1))*100

    # only Sat 
SINR_uav2_HAPS_sat = np.divide(SINR_uav2_HAPS_bs1, SINR_uav2_Sat_bs1)
x_Outage_sir_uav2_haps_from_sat = 10 * np.log10(SINR_uav2_HAPS_sat[SINR_uav2_HAPS_sat < 1])
sinr_outage_uav2_haps_from_sat_percent = (len(x_Outage_sir_uav2_haps_from_sat)/len(SINR_uav2_HAPS_sat))*100


#%% ARIMA Predictors

#data = yi2[0:300].astype(float)
#data = y3[0:40].astype(float)
#data = yi1[0:40].astype(float)
#data = yi1_uav3[40:80].astype(float)
#data = UEs_BSs1_0[0:50, 9].astype(float)
data = y2.astype(float)
timee = np.arange(0, len(data)) 

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# Split the data into training and testing sets
train_size = int(len(data) * 0.9)
train, test = data[:train_size+1], data[train_size:]

# Fit an ARIMA model
order = (2, 2, 6)  # ARIMA(p, d, q) order
model = ARIMA(train, order=order)
model_fit = model.fit()

# Make predictions on the test set
predictions = model_fit.forecast(steps=len(test))

# Evaluate the model performance
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')
# Plot the predicted values against the actual values
plt.plot(timee[:train_size+1],train, label='Train')
plt.plot(timee[train_size:],test, label='Test')
plt.plot(timee[train_size:],predictions, label='Predictions', linestyle='dashed')
plt.title(f'ARIMA Model Predictions\nOrder: {order}\nMSE: {mse:.2f} for P_rx of UE9 from BS1')
plt.xlabel('Time')
plt.ylabel('Data')
plt.legend()
plt.show()

####################
import matplotlib.pyplot as plt

# Index of the last BaseStation
last_bs_index = len(BaseStation_positions_time_now) - 1

# Create a 2D plot
plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
# Plot the last BaseStation
last_bs = BaseStation_positions_time_now[last_bs_index]
plt.scatter(last_bs[0]-last_bs[0], last_bs[1]- last_bs[1], label='Base Station', marker="d", s=150)

colors_list = plt.cm.viridis(np.linspace(0, 1, num_points))

# Plot UEs around the last BaseStation
for jjj in range(UE_positions_per_Bs_array.shape[1]):
    ue = UE_positions_per_Bs_array[last_bs_index, jjj, :]
    label = 'UE' if jjj == 0 else "_nolegend_"
    plt.scatter(ue[0]-last_bs[0], ue[1]-last_bs[1], marker='s', label=label, color=colors_list[jjj])

plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), prop={'size': 10})
plt.grid()
# Display the plot
plt.show()

"""
data = UEs_BSs1_0[:, 9].astype(float)
timee = np.arange(0, len(data)) 

# Split the data into training and testing sets
train_size = int(len(data) * 0.9)
train, test = data[:train_size+1], data[train_size:]
# Initialize lists to store MSE and RMSE
mse_list = []
rmse_list = []

# Initialize the training set for iterative forecasting
new_train = train.copy()

# Fit the initial ARIMA model
order = (3, 2, 1)
model = ARIMA(new_train, order=order)
model_fit = model.fit()

# Iterate through the test set
for i in range(len(test)):
    # Make one-step-ahead forecast
    predictions = model_fit.forecast(steps=1)

    # Calculate MSE and RMSE
    mse = mean_squared_error(test[i:i+1], predictions)
    rmse = np.sqrt(mse)

    # Store MSE and RMSE
    mse_list.append(mse)
    rmse_list.append(rmse)

    # Update the training set with the observed value from the test set
    new_train = np.append(new_train, test[i:i+1])

    # Update the ARIMA model
    new_model = ARIMA(new_train, order=order)
    new_model_fit = new_model.fit()

    # Update the model_fit for the next iteration
    model_fit = new_model_fit

# Plot the RMSE
plt.plot(rmse_list, marker='o')
plt.title(f'RMSE for Each Step Ahead Prediction \nOrder: {order}')
plt.xlabel('Step Ahead')
plt.ylabel('RMSE')
plt.show()
"""
"""
N = len(y2)
L = 10  # Window size for correlation
# Define auto-correlation function (ACF)
def acf(x):
    return np.exp(-0.1 * np.abs(x)/L)  # Example ACF, exponential decay

# Generate correlated Gaussian samples
cov_mat = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        cov_mat[i, j] = acf(np.abs(i - j))

# Cholesky decomposition to get correlated Gaussian samples
cholesky_decomp = np.linalg.cholesky(cov_mat)
gaussian_samples = np.exp(y2)
correlated_gaussian_samples = np.dot(cholesky_decomp, gaussian_samples)

# Transform correlated Gaussian samples to log-normal distribution
log_normal_samples = np.log(correlated_gaussian_samples)
#log_normal_samples = 10*np.log10(correlated_gaussian_samples)
#log_normal_samples = correlated_gaussian_samples
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(log_normal_samples,  label = 'correlated_samples',linewidth=2, color='blue')
plt.plot(y2,  label = 'Uncorrelated_samples',linewidth=2, color='red')
plt.title('Correlated Log-Normal Distributed Samples')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()


def auto_correlation(x, max_predict):
    auto_corr = np.zeros(max_predict)
    auto_corr[0] = np.corrcoef(x)
    for i in range(1, max_predict):
        auto_corr[i] = np.corrcoef(x[i:],x[:-i])[0,1]*(len(x)-i)/len(x)
    return auto_corr

plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
})
plt.plot(auto_correlation(log_normal_samples.astype(float), len(log_normal_samples)), linewidth=2, color='blue')
plt.xlabel('Lag')
plt.ylabel('Auto-correlation Value')
plt.title('Log-Normal Samples')
plt.grid(True)
plt.show()
"""

#%% Auxilary of lines which are commented now
# Lines 705 - 727
""" 
locals()[f'w_{i}'] = np.array(UE_ppp_Rx_power_df['P_rx (dBW)']) #auxilary variable
locals()[f'UEs_per_BSs_{i}'] = np.array([w.reshape(num_base_stations, num_points) for w in locals()[f'w_{i}']])
UEs_BSs1_0 = locals()[f'UEs_per_BSs_{i}'][:,0,:]
UEs_BSs2_0 = locals()[f'UEs_per_BSs_{i}'][:,1,:]
UEs_BSs3_0 = locals()[f'UEs_per_BSs_{i}'][:,2,:]
locals()[f'S_{i}'] = np.array(Interference_Sat_UEs_df['Interference (dBW)']) #auxilary variable 
locals()[f'Sat_on_UEs_{i}'] = np.array([np.array(w).reshape(num_base_stations, num_points) for w in locals()[f'S_{i}']])
UEs_of_BS1_Sat_Interferer_0 = locals()[f'Sat_on_UEs_{i}'][:,0,:]
UEs_of_BS2_Sat_Interferer_0 = locals()[f'Sat_on_UEs_{i}'][:,1,:]
UEs_of_BS3_Sat_Interferer_0 = locals()[f'Sat_on_UEs_{i}'][:,2,:]
locals()[f'H_{i}'] = np.array(Interference_HAP_UEs_df['Interference (dBW)']) #auxilary variable
locals()[f'HAP_on_UEs_{i}'] = np.array([np.array(w).reshape(num_base_stations, num_points) for w in locals()[f'H_{i}']])
UEs_of_BS1_HAPS_Interferer_0 = locals()[f'HAP_on_UEs_{i}'][:,0,:]
UEs_of_BS2_HAPS_Interferer_0 = locals()[f'HAP_on_UEs_{i}'][:,1,:]
UEs_of_BS3_HAPS_Interferer_0 = locals()[f'HAP_on_UEs_{i}'][:,2,:]
locals()[f'SIR_UEs_BS1_Sate_and_HAP_{i}'] = np.divide(10**(locals()[f'UEs_per_BSs_{i}'][:,0,:]/10), (10**(locals()[f'Sat_on_UEs_{i}'][:,0,:]/10))+(10**(locals()[f'HAP_on_UEs_{i}'][:,0,:]/10)))
locals()[f'SIR_UEs_BS2_Sate_and_HAP_{i}'] = np.divide(10**(locals()[f'UEs_per_BSs_{i}'][:,1,:]/10), (10**(locals()[f'Sat_on_UEs_{i}'][:,1,:]/10))+(10**(locals()[f'HAP_on_UEs_{i}'][:,1,:]/10)))
locals()[f'SIR_UEs_BS3_Sate_{i}'] = np.divide(10**(locals()[f'UEs_per_BSs_{i}'][:,2,:]/10), (10**(locals()[f'Sat_on_UEs_{i}'][:,2,:]/10)))
SIR_UEs_BS1_Sate_and_HAP_0 = locals()[f'SIR_UEs_BS1_Sate_and_HAP_{i}'] 
SIR_UEs_BS2_Sate_and_HAP_0 = locals()[f'SIR_UEs_BS2_Sate_and_HAP_{i}'] 
SIR_UEs_BS3_Sate_0 = locals()[f'SIR_UEs_BS3_Sate_{i}']
"""
