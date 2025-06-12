#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 18:42:51 2025

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
from LEO_satellite_cell_creation import HexagonGrid
from RX_power_calc import Rx_power
from HAPS_trajec_class import HAPS_trajectory
from Air_2_Ground_fading_channel import Air_Fading_channel
from Terresterial_Object import terresterial_network
from Satellite_fading_channel import Satellite_Fading_channel
from shadowing_temporally_correlated_AR import ShadowingFading
from Traffic_models import TrafficModel
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
    f = 2.0e9;
    satellite_EIRP_density, satellite_Tx_max_Gain, satellite_3dB_beamwidth, satellite_beam_diameter,  max_Bandwidth_per_beam = Satellite_communication_parameter().parameters(f, 'S', 'DL')
    satellite_EIRP_total = 10*np.log10((10**(satellite_EIRP_density/10))*max_Bandwidth_per_beam)
    A_z = 1*10**(-1) #Based on the figure 4 of the document ITU-R P.676-13 other values for 30 GHz is 0.2 and 5 GHz is 0.04 in dB
    G_max_Rx = 4 #dBi
    noise_figure_db = 7 # in dB
    temperature_k = 25 + 273.15 # in Kelvin      
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
    base_lat, base_lon = 53.110987, 8.851239
    radius_km = satellite_beam_diameter / 2
    # Create a hexagonal grid
    hex_grid = HexagonGrid(base_lat, base_lon, radius_km)
    satellite_cell_vertices = hex_grid.get_vertices()
    satellite_cell_vertices_in_wgs84 = hex_grid.convert_vertices_to_wgs84(satellite_cell_vertices)
    # Plot just the hexagon
    hex_grid.plot_hexagon()
    DF = LG.simulateConstellation(LEOs, groundstation, 20, time1, time2, ts = None, safetyMargin = 0)
    DF2 = DF.reset_index()
    #%% Walker constellation analysis
    
    # Calculate total visibility per satellite
    total_visibility = DF2.groupby('Satellite')['Visibility'].sum()

    # Convert to minutes
    total_visibility_minutes = total_visibility.dt.total_seconds() / 60
    # Get the first rise time per satellite
    first_rise = DF2.groupby('Satellite')['Rise'].min()

    # Sort satellites by first rise time
    satellite_order = first_rise.sort_values().index

    # Reorder the total visibility data
    total_visibility_minutes = total_visibility_minutes.loc[satellite_order]
    plt.figure(figsize=(7, 6), dpi = 1000)
    plt.bar(total_visibility_minutes.index, total_visibility_minutes.values)
    plt.xlabel('Satellite Name')
    plt.ylabel('Total Visibility Duration (minutes)')
    plt.title('Satellite Visibility Durations')
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.tight_layout()
    plt.show()
    
    # Solution 1: Show only every nth satellite label
    plt.figure(figsize=(3.6, 3), dpi=1000)
    plt.bar(total_visibility_minutes.index, total_visibility_minutes.values)
    plt.xlabel('Satellite Name')
    plt.ylabel('Total Visibility Duration (minutes)')
    #plt.title('Satellite Visibility Durations')
    
    # Show only every 10th satellite label
    n = 10  # Show every 10th label
    satellite_names = total_visibility_minutes.index
    tick_positions = range(len(satellite_names))
    plt.xticks(tick_positions[::n], satellite_names[::n], rotation=45, ha='right')
    plt.tight_layout()
    # Create output directory if it doesn't exist
    import os
    import tikzplotlib
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as TikZ using tikzplotlib
    tikz_filename = os.path.join(output_dir, 'satellite_visibility.tikz')
    tikzplotlib.save(tikz_filename)
    
    # Save as PGF
    pgf_filename = os.path.join(output_dir, 'satellite_visibility.pgf')
    plt.savefig(pgf_filename, 
                backend='pgf',
                bbox_inches='tight',
                dpi=1024)
    plt.show()

    # Solution 6: IEEE column-width optimized version
    plt.figure(figsize=(3.5, 3), dpi=300)  # IEEE single column width
    plt.bar(range(len(total_visibility_minutes)), total_visibility_minutes.values, width=0.8)
    plt.xlabel('Satellite Index', fontsize=9)
    plt.ylabel('Visibility Duration (min)', fontsize=9)
    plt.title('Satellite Visibility Durations', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.show()
    
    # Solution 7: Grouped visualization (if satellites belong to planes)
    # Assuming you want to show average visibility per plane
    satellite_plane_mapping = {}
    for i, sat_name in enumerate(total_visibility_minutes.index):
        plane_num = i // (numSat // numPlanes)  # Assuming equal distribution
        satellite_plane_mapping[sat_name] = f'Plane {plane_num}'
    
    # Create DataFrame for easier manipulation
    visibility_df = pd.DataFrame({
        'Satellite': total_visibility_minutes.index,
        'Visibility': total_visibility_minutes.values,
        'Plane': [satellite_plane_mapping[sat] for sat in total_visibility_minutes.index]
    })
    
    # Plot average visibility per plane
    plane_avg = visibility_df.groupby('Plane')['Visibility'].mean()
    plt.figure(figsize=(7, 6), dpi=1000)
    plt.bar(plane_avg.index, plane_avg.values)
    plt.xlabel('Satellite Plane')
    plt.ylabel('Average Visibility Duration (minutes)')
    plt.title('Average Satellite Visibility Duration by Plane')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    
    #%% All Data Frames initializations except satellite positions:
    sat_position_df = pd.DataFrame(columns=['Satellite ID', 'Time', 'Sat Position (km)', 'GS Position (km)', 'Distance from Earth Surface (km)', 'distance to GS (km)', 'Elevation Angle (degree)', 'satellite velocity (km)', 'v_LOS (kmps)'])
    sat_orbital_parameters_df = pd.DataFrame(columns=['Satellite ID', 'Time', 'Theta_el_sat_see_vsat', 'Theta_az_sat_see_vsat', 'theta_el_vsat_see_sat', 'theta_az_vsat_see_sat'])
    Satellite_channel_time_series_df = pd.DataFrame(columns=['Satellite ID', 'Time', 'Elevation Angle (degree)', 'small scale fading-channel', 'large scale shadowing'])
    Satellite_P_Rx_data_set = pd.DataFrame(columns=['Satellite ID', 'Time', 'Elevation Angle (degree)','Distance (km)', 'LoS Prob (%)', 'P_Rx_fspl_ShF_SSF (dBW)', 'P_rx_at_User (dBW)', 'SNR (dB)', 'shadowedrician_ssf', 'shadowing_lsf'])
    Interference_on_User_satellite = pd.DataFrame(columns=['Satellite ID', 'Time', 'HAPS_rx', 'BaseStation1_rx','BaseStation2_rx', 'BaseStation3_rx', 'Satellite_rx', 'SINR'])
    #%% LEO satellites' Fading channel initialization 
    # satellite user generation
    User_sat_initial_location = hex_grid.Point_initial_location_generation(base_lat, base_lon, 5)
    generation_intervals = 1 # seconds. 
    number_samples_nakagami = generation_intervals * 1000 #miliseconds; for microseconds multiply by 10e6
    num_samples_rician = 100 # in miliseconds
    fs_initial = 10000
    N = 256
    satellite_fading_channel = Satellite_Fading_channel(number_samples_nakagami, num_samples_rician, fs_initial, N)
    large_scale_shadowing_inetrval = 10 # in seconds. based on how many seconds to chnage 5 degree of elevation angle 
    large_scale_shadowing_num_samples = large_scale_shadowing_inetrval * 1000
    Shadowing = ShadowingFading(tau = 10, N = large_scale_shadowing_num_samples)

    #%%
    """ ################################################# Parameters intialization for UAV, drones and HAPs ################################################################ """
    
    ############################################## First assign the environemnt of propagation and frequecny ################################
    ########## HASP Tarjectory ###############
    HAPS_height = 10
    velocity_haps = 25.0 # in km/hour
    radius_haps = 6 # in km
    HAPS_initial_location = hex_grid.Point_initial_location_generation(base_lat, base_lon, 10)
    HAPS1 =  HAPS_trajectory(velocity_haps, radius_haps, time_interval=1)
    angular_velocity_rad_haps, number_step_haps = HAPS1.get_values()
    shiftak0_haps = 0
    P_HAPS1_tx = 6 # in dB
    G_HAPS1_tx = 8 + 10*np.log10(16)
    HAPS_generation_intervals = 3 # seconds. 
    HAPS_number_samples_nakagami = HAPS_generation_intervals * 1000 #miliseconds; for microseconds multiply by 10e6
    HAPS_num_samples_rician = 300 # in miliseconds
    HAPS_fading_channel = Air_Fading_channel(HAPS_number_samples_nakagami, HAPS_num_samples_rician, fs_initial, N)
    HAPS_large_scale_shadowing_inetrval = 10 # in seconds. 
    HAPS_large_scale_shadowing_num_samples = HAPS_large_scale_shadowing_inetrval * 1000
    HAPS_Shadowing = ShadowingFading(tau = 30, N = HAPS_large_scale_shadowing_num_samples)
    #%%
    """ ################################################# Parameters intialization for Terresterial network and objects ######################################################## """
    ############################################## First assign the environemnt of propagation and frequecny ################################
    Terresterial = terresterial_network(f, 'Suburban')
    radius_Terresterial = 20 # The value is in km
    ################################### Deployemnet scenarios ###################################
    # scenario 1_1: Teresterial objects UEs are distributed over the area as Poisson Point Process to be served by BaseStations 
    lamb = 10 # This the lambda of PPP
    radius_per_each_BS = 10    
    ## scenario 1_2: Teresterial BaseStation are distributed over the area initially as a Binomial Point Process and then remained fixed 
    num_base_stations = 3
    [center_x, center_y, center_z] = groundstation.itrs_xyz.km
    BaseStation_position0 = Terresterial.generate_base_station_positions(center_x, center_y, radius_Terresterial, num_base_stations, center_z, radius_per_each_BS, 1000)
    BaseStation_position0_array = np.array(BaseStation_position0)
    BaseStation_latlon_coordinates = Terresterial.cartesian_to_latlon(np.array([BaseStation_position0]).reshape(-1,3))
    BaseStation_latlon_coordinates = np.array([BaseStation_latlon_coordinates]).reshape(-1,2)
    BaseStations_skyfield_positions = Terresterial.skyfield_position_for_BaseStations(BaseStation_latlon_coordinates)
    BS1 = BaseStations_skyfield_positions[0]
    BS2 = BaseStations_skyfield_positions[1] 
    BS3 = BaseStations_skyfield_positions[-1]
    height_BS = 35  # this value is in meter
    P_BSs_tx = 16 # in dB
    G_BSs_tx = 8 + 10*np.log10(16) # dBi; This value is based on 3gpp TR 38.901 page 23 Table 7.3-1: Radiation power pattern of a single antenna element. The BS antenna is modelled by a uniform rectangular panel array, comprising MgNg panels
    BSs_generation_intervals = 10 # seconds
    #%% Traffic patterns
    # Initialize traffic models for all satellites (CBR)
    satellite_traffic_models = {}
    for sat_id in range(1, numSat + 1):
        satellite_traffic_models[f'Sat {sat_id}'] = TrafficModel(
            model_type='CBR',
            packet_size=1000,
            packet_rate=10
        )
        satellite_traffic_models[f'Sat {sat_id}'].generate_packets(seconds_difference_time)
    
    # HAPS Traffic Model (Poisson)
    haps_traffic = TrafficModel(
        model_type='Poisson',
        avg_packet_rate=10,
        packet_size_mean=1000,
        packet_size_std=200
    )
    haps_traffic.generate_packets(seconds_difference_time)
    
    # Base Station 1 Traffic Model (Poisson) - High traffic
    bs1_traffic = TrafficModel(
        model_type='Poisson',
        avg_packet_rate=10,  # 20 packets/second
        packet_size_mean=1000,
        packet_size_std=200
    )
    bs1_traffic.generate_packets(seconds_difference_time)
    
    # Base Station 2 Traffic Model (Poisson) - Moderate traffic
    bs2_traffic = TrafficModel(
        model_type='Poisson',
        avg_packet_rate=8,  # 10 packets/second
        packet_size_mean=1000,
        packet_size_std=200
    )
    bs2_traffic.generate_packets(seconds_difference_time)
    
    # Base Station 3 Traffic Model (Poisson) - Low traffic
    bs3_traffic = TrafficModel(
        model_type='Poisson',
        avg_packet_rate=5,   # 5 packets/second
        packet_size_mean=1000,
        packet_size_std=200
    )
    bs3_traffic.generate_packets(seconds_difference_time)
    
    
    # Test CBR
    cbr = TrafficModel('CBR', packet_size=1000, packet_rate=10)
    cbr_packets = cbr.generate_packets(1.0)
    print("CBR packets:", cbr_packets)
    
    # Test Poisson
    poisson = TrafficModel('Poisson', avg_packet_rate=5, packet_size_mean=1000, packet_size_std=200)
    poisson_packets = poisson.generate_packets(1.0)
    print("Poisson packets:", poisson_packets)
    
    # Test Bursty
    bursty = TrafficModel('Bursty', on_duration=0.5, off_duration=0.5, packet_rate_on=20, packet_size=1000)
    bursty_packets = bursty.generate_packets(1.0)
    print("Bursty packets:", bursty_packets)
    
    
    #%% Mobility loop for all elements
    for i in range(0,len(DF2)):
        Sat_ID = DF2.iloc[i,0]
        Sat_ID_int = int(Sat_ID[3:])
        t_rise  = DF2.iloc[i,1]
        t_set = DF2.iloc[i,2]
        t_rise_now = ts.utc(t_rise.year, t_rise.month, t_rise.day, t_rise.hour, t_rise.minute, t_rise.second)
        t_set_now = ts.utc(t_set.year, t_set.month, t_set.day, t_set.hour, t_set.minute, t_set.second)
        visibility_sec = LG.difference_time_in_seconds(t_rise_now, t_set_now)
        visibility_millisec = visibility_sec * 1000
        # Find all visible satellites at t_rise_now
        visible_sats = []
        max_elevation = -90
        serving_sat_id = None
        # Inside the loop over j:
        for j in range(len(DF2)):
            other_sat_id = DF2.iloc[j, 0]
            other_t_rise_pd = DF2.iloc[j, 1]  # pandas Timestamp
            other_t_set_pd = DF2.iloc[j, 2]    # pandas Timestamp
            
            # Convert pandas Timestamps to datetime objects
            other_t_rise_dt = other_t_rise_pd.to_pydatetime()
            other_t_set_dt = other_t_set_pd.to_pydatetime()
            
            # Convert datetime objects to skyfield Time objects
            other_t_rise_sky = ts.utc(
                other_t_rise_dt.year, other_t_rise_dt.month, other_t_rise_dt.day,
                other_t_rise_dt.hour, other_t_rise_dt.minute, other_t_rise_dt.second
            )
            other_t_set_sky = ts.utc(
                other_t_set_dt.year, other_t_set_dt.month, other_t_set_dt.day,
                other_t_set_dt.hour, other_t_set_dt.minute, other_t_set_dt.second
            )
            
            # Check if current time is within visibility window using skyfield Time objects
            if other_t_rise_sky.utc_datetime() <= t_rise_now.utc_datetime():
                if t_rise_now.utc_datetime() <= other_t_set_sky.utc_datetime():
                    other_sat_id_int = int(other_sat_id[3:])
                    other_pos = LEOs[other_sat_id_int - 1].at(t_rise_now)
                    elevation = LG.elevation_angel_calculator(other_pos.xyz.km, groundstation.at(t_rise_now).position.km)
                    visible_sats.append((other_sat_id, other_sat_id_int, other_pos, elevation))
                    if elevation > max_elevation:
                        max_elevation = elevation
                        serving_sat_id = other_sat_id
        
        # In oder to calculate in miliseconds you need to convert visibility_sec to miliseconds by: visibility_sec*1000 and then in 
        # datetime.timedelta(seconds=i1) write instead of seconds as microseconds = i1*1000 because it doesn't accept miliseconds 
        
        """ ########################### Second loop over visibility duration of each satellite ############################ """
        
        for i1 in range(0,visibility_millisec):

            time_now = t_rise_now+datetime.timedelta(microseconds=i1*1000)
            satellite_cell_vertice_at_time_now = satellite_cell_vertices_in_wgs84.at(time_now)
            satellite_cell_vertice_positions = satellite_cell_vertice_at_time_now.position.km
            position_LEO = LEOs[Sat_ID_int-1].at(time_now)
            LEO_velocity = LEOs[Sat_ID_int-1].at(time_now).velocity
            LEO_velocity_km_per_sec = LEO_velocity.km_per_s
            line_of_sight_vector_from_GS_to_satellite = groundstation.at(time_now).position.km - position_LEO.xyz.km
            r_hat = line_of_sight_vector_from_GS_to_satellite / np.linalg.norm(line_of_sight_vector_from_GS_to_satellite)
            v_LOS = np.dot(LEO_velocity_km_per_sec, r_hat)
            
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
            
            
            # Get ground station position in ECEF
            gs_pos_ecef = groundstation.at(time_now).position.km
            # Get ground station latitude and longitude
            gs_lat = groundstation.latitude.radians
            gs_lon = groundstation.longitude.radians
            
            # Create rotation matrix from ECEF to ground station local horizon (East-North-Up)
            R_ecef_to_enu = np.array([
                [-np.sin(gs_lon), np.cos(gs_lon), 0],
                [-np.sin(gs_lat)*np.cos(gs_lon), -np.sin(gs_lat)*np.sin(gs_lon), np.cos(gs_lat)],
                [np.cos(gs_lat)*np.cos(gs_lon), np.cos(gs_lat)*np.sin(gs_lon), np.sin(gs_lat)]
            ])
            
            # Vector from ground station to satellite
            r_gs_to_sat = position_LEO.xyz.km - gs_pos_ecef
            
            # Transform to local horizon coordinates
            r_gs_to_sat_enu = R_ecef_to_enu @ r_gs_to_sat
            r_enu_norm = np.linalg.norm(r_gs_to_sat_enu)
            east, north, up = r_gs_to_sat_enu
            
            # Calculate elevation and azimuth
            theta_el_vsat_see_sat = np.arcsin(up / r_enu_norm)
            theta_az_vsat_see_sat = np.arctan2(east, north)
            # Print results
            print(f"Relative position of LEO to VSAT in global frame: {r_gs_to_sat_enu}")
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
                'Elevation Angle (degree)': [elevation_angle],
                'satellite velocity (km)':[LEO_velocity_km_per_sec],
                'v_LOS (kmps)':[v_LOS]
            })
            
            new_row_sat_orbital_param = pd.DataFrame({
                'Satellite ID': [Sat_ID],
                'Time': [time_now.utc_datetime()],
                'Theta_el_sat_see_vsat':[Theta_el_sat_see_vsat], 
                'Theta_az_sat_see_vsat': [Theta_az_sat_see_vsat], 
                'theta_el_vsat_see_sat':[theta_el_vsat_see_sat], 
                'theta_az_vsat_see_sat':[theta_az_vsat_see_sat]
            })
            
            # Concatenate the new row DataFrame with the existing DataFrame
            sat_position_df = pd.concat([sat_position_df, new_row_sat_position], ignore_index=True)
            sat_orbital_parameters_df = pd.concat([sat_orbital_parameters_df, new_row_sat_orbital_param], ignore_index=True)
            
            # Assign the propagation environemnt for line of sight probability claculating 
            lOS_prob = Rx_power().LOS_prob_calc(elevation_angle, 'Sub_Urban')
            
            # calculate the Shadowed-Rician fading channel temporally correlated as:
                # we generate the channel for each generation_intervals seconds since the change of elevation angle each generation_intervals seconds is significant
                # we generate per each second 1000 samples means TTI is 1 ms (10e6 samples if TTI is in microseconds)
                # Here since TTI is 1 ms, we feed at each i2 one sample out of generated channels 
            if i1 % generation_intervals*1000 == 0:
                print(f"Generating samples at i={i}, i1={i1}")
                shadowed_rician_channel_samples_interval = satellite_fading_channel.run_simulation(elevation_angle, v_LOS, f, distance_GS_sat)
            if i1 % large_scale_shadowing_inetrval*1000 == 0:
                shadowing_samples_interval = Shadowing.SF_LOS_calc(elevation_angle, 'LOS', 'SBand')
                print(f"Generating LS sahdowing samples at i={i}, i1={i1}")

            start_idx_small_scale_fading = i1 % generation_intervals*1000
            end_idx_small_scale_fading = start_idx_small_scale_fading + 1
            
            start_idx_large_scale_fading = i1 % large_scale_shadowing_inetrval*1000
            end_idx_large_scale_fading = start_idx_large_scale_fading + 1
       
            if end_idx_small_scale_fading <= len(shadowed_rician_channel_samples_interval):
               Satellite_Channel_fading_samples = shadowed_rician_channel_samples_interval[start_idx_small_scale_fading:end_idx_small_scale_fading]
            else:
               print(f"Index out of range at i={i}, i1={i1}")
               
            if end_idx_large_scale_fading <= len(shadowing_samples_interval):
               Satellite_Shadowing_samples = shadowing_samples_interval[start_idx_large_scale_fading:end_idx_large_scale_fading]
            else:
               print(f"Index out of range at i={i}, i1={i1}")

            
            # calculate only the fspl as function of distance and frequecny
            fspl_solo = Rx_power().FSPl_only(f, distance_GS_sat)
            
            atmospheric_loss = Rx_power().atmospheric_att(A_z, elevation_angle)
                                    
            satellite_packets = satellite_traffic_models[Sat_ID].get_packets_at_time(i1, time_window=0.001)            
            #P_received_fspl_ShF_SSF = 0
            #if satellite_packets:
            P_received_fspl_ShF_SSF = (satellite_EIRP_total + G_max_Rx + 20*np.log10(np.abs(Satellite_Channel_fading_samples))) - fspl_solo - atmospheric_loss - Satellite_Shadowing_samples

            User_satellite_initial_loc = User_sat_initial_location.at(time_now).position.km
            
            distance_User_sat_center_beam = LG.distance(groundstation.at(time_now).position.km, User_satellite_initial_loc)
                    
            P_rx_User_sat = P_received_fspl_ShF_SSF - 10 * 2 * np.log10(1 - distance_User_sat_center_beam/satellite_beam_diameter/2)

            Noise_power = Rx_power().Noise_power_with_NoiseFigure(noise_figure_db, max_Bandwidth_per_beam, temperature_k)  
            
            new_row_Satellite_channel_time_series_df = pd.DataFrame({
                'Satellite ID': [Sat_ID],
                'Time': [time_now.utc_datetime()],
                'Elevation Angle (degree)': [elevation_angle],
                'small scale fading-channel': [Satellite_Channel_fading_samples],
                'large scale shadowing': [Satellite_Shadowing_samples]
                })
            new_row_P_Rx_data_set = pd.DataFrame({
                'Satellite ID': [Sat_ID],
                'Time': [time_now.utc_datetime()],
                'Elevation Angle (degree)': [elevation_angle],
                'Distance (km)': [distance_GS_sat],
                'LoS Prob (%)': [lOS_prob],
                'P_Rx_fspl_ShF_SSF (dBW)': [P_received_fspl_ShF_SSF],
                'P_rx_at_User (dBW)':[P_rx_User_sat],
                'SNR (dB)': [P_rx_User_sat - Noise_power],
                'shadowedrician_ssf' : [Satellite_Channel_fading_samples], 
                'shadowing_lsf' : [Satellite_Shadowing_samples]
                })
            Satellite_P_Rx_data_set = pd.concat([Satellite_P_Rx_data_set, new_row_P_Rx_data_set], ignore_index=True)
            # now for HAPS
            shiftak_haps = (i1 + shiftak0_haps) % number_step_haps
            step_haps = shiftak_haps + 1
            HAPS1_initial_loc = HAPS_initial_location.at(time_now).position.km
            HAPS1_position = HAPS1.simulate_circular_trajectory(HAPS1_initial_loc, 10, step_haps, position_GS)
            elevation_angle_User_sat_HAPS = 90 - LG.elevation_angel_calculator(HAPS1_position[0,:3].flatten(), User_satellite_initial_loc)
            distance_User_sat_to_HAPS = LG.distance(HAPS1_position[0,:3].flatten(), User_satellite_initial_loc)
            if i1 % HAPS_generation_intervals*1000 == 0:
                print(f"Generating HAPS channel samples at i={i}, i1={i1}")
                HAPS_shadowed_rician_channel_samples_interval = HAPS_fading_channel.run_simulation(elevation_angle_User_sat_HAPS, velocity_haps, f, distance_User_sat_to_HAPS)
            if i1 % HAPS_large_scale_shadowing_inetrval*1000 == 0:
                HAPS_shadowing_samples_interval = Shadowing.SF_LOS_calc(elevation_angle_User_sat_HAPS, 'LOS', 'SBand')
                print(f"Generating LS sahdowing samples at i={i}, i1={i1}")

            HAPS_start_idx_small_scale_fading = i1 % HAPS_generation_intervals*1000
            HAPS_end_idx_small_scale_fading = HAPS_start_idx_small_scale_fading + 1
            
            HAPS_start_idx_large_scale_fading = i1 % HAPS_large_scale_shadowing_inetrval*1000
            HAPS_end_idx_large_scale_fading = HAPS_start_idx_large_scale_fading + 1
       
            if HAPS_end_idx_small_scale_fading <= len(HAPS_shadowed_rician_channel_samples_interval):
               HAPS_Channel_fading_samples = HAPS_shadowed_rician_channel_samples_interval[HAPS_start_idx_small_scale_fading:HAPS_end_idx_small_scale_fading]
            else:
               print(f"Index out of range at i={i}, i1={i1}")
               
            if HAPS_end_idx_large_scale_fading <= len(HAPS_shadowing_samples_interval):
               HAPS_Shadowing_samples = HAPS_shadowing_samples_interval[HAPS_start_idx_large_scale_fading:HAPS_end_idx_large_scale_fading]
            else:
               print(f"Index out of range at i={i}, i1={i1}")
            
            fspl_HAPS_User_satellite = Rx_power().FSPl_only(f, distance_User_sat_to_HAPS)
            haps_packets = haps_traffic.get_packets_at_time(i1, time_window=0.001)
            haps_rx_power = 0
            if haps_packets:
                for packet_size in haps_packets:
                    interference_power = (P_HAPS1_tx + G_HAPS1_tx + 20*np.log10(np.abs(HAPS_Channel_fading_samples))) - fspl_HAPS_User_satellite - HAPS_Shadowing_samples
                    haps_rx_power += 10 ** (interference_power / 10)  # Convert to mW and sum

            # Base Stations
            # Base Stations
            BaseStation_positions_time_now = Terresterial.get_base_station_positions_at_time_now(time_now, BaseStations_skyfield_positions, height_BS)
            bs_interference = [0, 0, 0]
            
            # Retrieve packets for each base station
            bs1_packets = bs1_traffic.get_packets_at_time(i1, time_window=0.001)
            bs2_packets = bs2_traffic.get_packets_at_time(i1, time_window=0.001)
            bs3_packets = bs3_traffic.get_packets_at_time(i1, time_window=0.001)
            
            for iii, bs_pos in enumerate(BaseStation_positions_time_now):
                distance_user_to_bs = LG.distance(bs_pos, User_satellite_initial_loc)
                Pathloss_bs_User_LoS, Pathloss_bs_User_nLoS = Terresterial.pathloss_calculator_up_to_7GHz(User_satellite_initial_loc, bs_pos, 35, 1.5)
                Pathloss_bs_User = max(Pathloss_bs_User_LoS, Pathloss_bs_User_nLoS)
                
                # Generate Rician fading channel
                if i1 % BSs_generation_intervals*1000 == 0:
                    print(f"Generating BS channel samples at i={i}, i1={i1}")
                    Rician_channel_bs_to_User = Terresterial.rician_fading_accurate(10000, 1, fs_initial, 10, np.pi/4)
                
                BSs_start_idx_small_scale_fading = i1 % (BSs_generation_intervals * 1000)
                BSs_end_idx_small_scale_fading = BSs_start_idx_small_scale_fading + 1
                
                if BSs_end_idx_small_scale_fading <= len(Rician_channel_bs_to_User):
                    Rician_channel_bs_to_User_samples = Rician_channel_bs_to_User[BSs_start_idx_small_scale_fading:BSs_end_idx_small_scale_fading]
                else:
                    print(f"Index out of range at i={i}, i1={i1}")
                    Rician_channel_bs_to_User_samples = [0]
                
                # Calculate interference for each base station independently
                if iii == 0 and bs1_packets:
                    for packet_size in bs1_packets:
                        interference_power = (P_BSs_tx + G_BSs_tx + 20*np.log10(np.abs(Rician_channel_bs_to_User_samples))) - Pathloss_bs_User
                        bs_interference[iii] += 10 ** (interference_power / 10)  # Convert to W
                if iii == 1 and bs2_packets:
                    for packet_size in bs2_packets:
                        interference_power = (P_BSs_tx + G_BSs_tx + 20*np.log10(np.abs(Rician_channel_bs_to_User_samples))) - Pathloss_bs_User
                        bs_interference[iii] += 10 ** (interference_power / 10)  # Convert to W
                if iii == 2 and bs3_packets:
                    for packet_size in bs3_packets:
                        interference_power = (P_BSs_tx + G_BSs_tx + 20*np.log10(np.abs(Rician_channel_bs_to_User_samples))) - Pathloss_bs_User
                        bs_interference[iii] += 10 ** (interference_power / 10)  # Convert to W
                        
            SINR = (10**(P_rx_User_sat/10)) / (10**(Noise_power/10) + haps_rx_power + bs_interference[0] + bs_interference[1] + bs_interference[2])
            # Update Interference DataFrame
            new_row_interference = pd.DataFrame({
                'Satellite ID': [serving_sat_id if serving_sat_id else Sat_ID],
                'Time': [time_now.utc_datetime()],
                'HAPS_rx': [10 * np.log10(haps_rx_power) if haps_rx_power > 0 else Noise_power],
                'BaseStation1_rx': [10 * np.log10(bs_interference[0]) if bs_interference[0] > 0 else Noise_power],
                'BaseStation2_rx': [10 * np.log10(bs_interference[1]) if bs_interference[1] > 0 else Noise_power],
                'BaseStation3_rx': [10 * np.log10(bs_interference[2]) if bs_interference[2] > 0 else Noise_power],
                'Satellite_rx': [P_rx_User_sat if 10**(P_rx_User_sat/10) > 0 else Noise_power],
                'SINR (dB)': [10*np.log10(SINR)]
            })
            Interference_on_User_satellite = pd.concat([Interference_on_User_satellite, new_row_interference], ignore_index=True)
            
            print(i1)
            shiftak0_haps = (shiftak0_haps + visibility_sec) % number_step_haps
            shiftak_haps = (visibility_sec + shiftak0_haps) % number_step_haps
        print(i)
        # Create a 2D plot
        plt.figure(figsize=(8, 6))
        for ver in range(len(satellite_cell_vertices)):
            plt.plot(satellite_cell_vertice_positions[0,ver], satellite_cell_vertice_positions[1,ver], label='cell border', c='blue', marker = 'o')
        plt.scatter(position_GS[0], position_GS[1], label='Cell center', c='red', marker='o')
        plt.scatter(User_satellite_initial_loc[0], User_satellite_initial_loc[1], label='Sat. User', c='c', marker = 'o')
        for iii, bs in enumerate(BaseStation_positions_time_now):
            plt.scatter(bs[0], bs[1], label=f'Base Station {iii + 1}', marker="d", s=150)
        plt.scatter(HAPS1_position[:,0], HAPS1_position[:,1], label = 'HAP', c = 'm', marker = "h")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='upper right',bbox_to_anchor=(1.1, 1), prop={'size': 7})
        plt.grid()
        plt.show()
#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tikzplotlib

# Define noise floor value (from your simulation)
NOISE_FLOOR = -122.086  # dB

# Create a copy of the interference dataframe
df = Interference_on_User_satellite.copy()
df = df.iloc[:-1]

# Save the DataFrame as a Parquet file
for col in df.columns:
    if df[col].apply(lambda x: isinstance(x, np.ndarray)).any():
        df[col] = df[col].apply(
            lambda x: x.item() if isinstance(x, np.ndarray) and x.size == 1 else x
        )

df.to_parquet(file_path_to_save, index=False)
print(f"DataFrame saved to {file_path_to_save}")

# Function to detect interference presence
def detect_interference(row, source):
    # Assume scalar or ndarray input
    value = row[source]
    
    # If it's a NumPy array or list, reduce it to scalar (e.g., mean or any)
    if isinstance(value, (np.ndarray, list)):
        return np.any(np.abs(np.array(value) - NOISE_FLOOR) > 0.1)
    
    # Else, just scalar float
    return abs(value - NOISE_FLOOR) > 0.1


# Add columns indicating interference presence
df['HAPS_active'] = df.apply(lambda row: detect_interference(row, 'HAPS_rx'), axis=1)
df['BS1_active'] = df.apply(lambda row: detect_interference(row, 'BaseStation1_rx'), axis=1)
df['BS2_active'] = df.apply(lambda row: detect_interference(row, 'BaseStation2_rx'), axis=1)
df['BS3_active'] = df.apply(lambda row: detect_interference(row, 'BaseStation3_rx'), axis=1)
# Clean list values that wrap single bools
for col in ['HAPS_active', 'BS1_active', 'BS2_active', 'BS3_active']:
    df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)

# Create interference scenario classification
def classify_scenario(row):
    active_sources = []
    if row['HAPS_active']: active_sources.append('HAPS')
    if row['BS1_active']: active_sources.append('BS1')
    if row['BS2_active']: active_sources.append('BS2')
    if row['BS3_active']: active_sources.append('BS3')
    
    num_active = len(active_sources)
    
    if num_active == 0:
        return 'No Interference'
    elif num_active == 1:
        return f'{active_sources[0]} only'
    elif num_active == 2:
        return f'Two sources: {"+".join(active_sources)}'
    elif num_active == 3:
        return f'Three sources: {"+".join(active_sources)}'
    elif num_active == 4:
        return 'All four sources'
    return 'Unknown'

df['Scenario'] = df.apply(classify_scenario, axis=1)

# Create individual interference scenario dataframes
haps_only = df[df.apply(lambda row: row['HAPS_active'] and 
                        not row['BS1_active'] and 
                        not row['BS2_active'] and 
                        not row['BS3_active'], axis=1)]

bs1_only = df[df.apply(lambda row: not row['HAPS_active'] and 
                       row['BS1_active'] and 
                       not row['BS2_active'] and 
                       not row['BS3_active'], axis=1)]

bs2_only = df[df.apply(lambda row: not row['HAPS_active'] and 
                       not row['BS1_active'] and 
                       row['BS2_active'] and 
                       not row['BS3_active'], axis=1)]

bs3_only = df[df.apply(lambda row: not row['HAPS_active'] and 
                       not row['BS1_active'] and 
                       not row['BS2_active'] and 
                       row['BS3_active'], axis=1)]

# Create combined interference scenarios
multiple_interferers = df[df.apply(lambda row: 
                                   sum([row['HAPS_active'], row['BS1_active'], 
                                       row['BS2_active'], row['BS3_active']]) > 1, axis=1)]
# Filter the No Interference case where SINR ≈ SNR
no_interference = df[df['Scenario'] == 'No Interference']

# Function to plot empirical CDF with optional marker
def plot_empirical_cdf(data, label, color, ax, marker=None):
    if len(data) > 0:
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        ax.plot(
            sorted_data,
            yvals,
            label=label,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=5,
            markevery=20 if marker else None  # space out markers if present
        )
        return True
    return False

# Plot setup
plt.figure(figsize=(4.5, 3.2))
ax = plt.gca()
colors = plt.cm.tab10.colors

# Define marker styles for interference scenarios
markers = {
    'HAPS only': 'o',
    'BS1 only': '*',
    'BS2 only': '+',
    'BS3 only': 's',
    'SNR (No Interference)': None  # Continuous line
}

# Define scenarios and plot
scenarios = [
    (haps_only, 'HAPS only', colors[0]),
    (bs1_only, 'BS1 only', colors[1]),
    (bs2_only, 'BS2 only', colors[2]),
    (bs3_only, 'BS3 only', colors[3]),
    (no_interference, 'SNR (No Interference)', colors[4]),
]

for scenario, label, color in scenarios:
    if len(scenario) > 0:
        plot_empirical_cdf(
            data=scenario['SINR (dB)'],
            label=label,
            color=color,
            ax=ax,
            marker=markers[label]
        )

# Add plot decorations
plt.xlabel('SINR [dB]', fontsize=10)
plt.ylabel('Empirical Cumulative Distribution Function (ECDF)', fontsize=10)
#plt.title('Empirical CDF of SINR for Different Interference Scenarios', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=4)
plt.tight_layout()

# Save plot with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'sinr_cdf_analysis_{timestamp}.png', dpi=300)
plt.savefig(f'sinr_cdf_analysis_{timestamp}.pgf')

# Save as .tikz for LaTeX
tikzplotlib.save(f"sinr_cdf_analysis_{timestamp}.tikz")

plt.show()

# Additional analysis: Print statistics
print("\nInterference Scenario Statistics:")
print(f"Total samples: {len(df)}")
print(f"HAPS only: {len(haps_only)} samples ({len(haps_only)/len(df)*100:.1f}%)")
print(f"BS1 only: {len(bs1_only)} samples ({len(bs1_only)/len(df)*100:.1f}%)")
print(f"BS2 only: {len(bs2_only)} samples ({len(bs2_only)/len(df)*100:.1f}%)")
print(f"BS3 only: {len(bs3_only)} samples ({len(bs3_only)/len(df)*100:.1f}%)")
print(f"Multiple interferers: {len(multiple_interferers)} samples ({len(multiple_interferers)/len(df)*100:.1f}%)")



