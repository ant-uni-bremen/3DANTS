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
from HAPS_trajec_class import HAPS_trajectory
from fading_channel_sim import FadingSimulation
from Air_2_Ground_fading_channel import Fading_channel_air_to_Ground 
from shadowing_temporally_correlated_AR import ShadowingFading
import matplotlib.pyplot as plt
import pandas as pd
import copy
import os


from pyproj import Geod

def hexagon_vertices(center_lat, center_lon, radius_km):
    """Generate vertices for a hexagonal cell centered at (center_lat, center_lon) with given radius."""
    geod = Geod(ellps="WGS84")
    vertices = []
    for angle in range(0, 360, 60):  # 6 vertices, spaced by 60 degrees
        lon, lat, _ = geod.fwd(center_lon, center_lat, angle, radius_km * 1000)
        vertices.append((lat, lon))
    return vertices

def compute_new_hexagon_center(lat1, lon1, lat2, lon2, radius_km):
    """Calculate the center of the new hexagon based on the southern side."""
    geod = Geod(ellps="WGS84")
    
    # Compute midpoint of the southern side
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    
    # Compute the azimuth of the line passing through the two points
    fwd_azimuth, _, _ = geod.inv(lon1, lat1, lon2, lat2)
    
    # Perpendicular azimuth (add 90 degrees)
    #perpendicular_azimuth = fwd_azimuth - 270
    perpendicular_azimuth = fwd_azimuth + 90
    
    # Compute the center by moving 25 km along the perpendicular bisector
    center_lon, center_lat, _ = geod.fwd(mid_lon, mid_lat, perpendicular_azimuth, radius_km * 1000)
    
    return center_lat, center_lon

def vertices_sorting(vertices):
    return sorted(vertices, key=lambda x: x[0], reverse=True)

def sort_vertices_counterclockwise(vertices):
    # Calculate centroid
    center_lat = sum(v[0] for v in vertices) / len(vertices)
    center_lon = sum(v[1] for v in vertices) / len(vertices)
    
    # Function to calculate angle between point and horizontal axis
    def get_angle(point):
        return np.arctan2(point[0] - center_lat, point[1] - center_lon)
    
    # Sort vertices by angle
    sorted_vertices = sorted(vertices, key=get_angle, reverse=True)
    
    # Add first vertex at the end to close the polygon
    return sorted_vertices + [sorted_vertices[0]]

def north_edge(vertices):
    """Get the two points of the northern edge of the hexagon."""
    vertices_sorted = vertices_sorting(vertices)
    # The two highest latitude points define the northern edge
    northern_point1 = vertices_sorted[0]
    northern_point2 = vertices_sorted[1]
    return northern_point1, northern_point2



def plot_hexagon(vertices, center):
    """Plot hexagonal cell and its center."""
    vertices.append(vertices[0])  # Close the hexagon
    lats, lons = zip(*vertices)
    plt.figure(figsize=(8, 8))
    plt.plot(lons, lats, marker='o', label="Hexagon Edges")
    plt.scatter(center[1], center[0], color='red', label="Base Station")
    plt.scatter(north_lon, north_lat, color='blue', label="Northern Edge Point")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid()
    plt.show()
    



def calculate_midpoint(lat1, lon1, lat2, lon2):
    """Calculate the midpoint between two geographic points."""
    # Compute midpoint of the southern side
    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    return mid_lat, mid_lon

def overlapping_rhombus(bs_northern_edge, satellite_southern_edge, vertices):
    """
    Calculate the vertices of the overlapping rhombus.
    bs_northern_edge: Tuple (latitude, longitude) of the base station's northern edge point.
    satellite_southern_edge: Tuple (latitude, longitude) of the satellite's southern edge point.
    vertices: Sorted list of base station hexagon vertices.
    """
    # Calculate the midpoints of the legs adjacent to the northern edge
    mid1 = calculate_midpoint(vertices[0][0], vertices[0][1], vertices[1][0], vertices[1][1])
    mid2 = calculate_midpoint(vertices[0][0], vertices[0][1], vertices[2][0], vertices[2][1])
    
    # Form the rhombus vertices
    rhombus_vertices = [
        bs_northern_edge,
        satellite_southern_edge,
        mid1,
        mid2
    ]
    return rhombus_vertices

def plot_two_hexagons(vertices1, center1, vertices2, center2):
    """Plot the original and new hexagonal cells."""
    vertices1.append(vertices1[0])  # Close the first hexagon
    vertices2.append(vertices2[0])  # Close the second hexagon

    lats1, lons1 = zip(*vertices1)
    lats2, lons2 = zip(*vertices2)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(lons1, lats1, marker='o', label="Basestation coverage cell ($r_{bs}$ = 15 km)")
    ax.scatter(center1[1], center1[0], color='red', label="Base Station Center")
    ax.plot(lons2, lats2, marker='o', label="LEO Satellite coverage cell ($r_{sat}$ = 25 km)")
    ax.scatter(center2[1], center2[0], color='blue', label="Center of Satellite Beam")
    
    """Plot the overlapping rhombus."""
    #rhombus_vertices.append(rhombus_vertices[0])  # Close the rhombus
    #lats, lons = zip(*rhombus_vertices)
    #plt.plot(lons, lats, marker='o', color='green', label="Overlapping Rhombus")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid()
    ax.set_title("Hexagonal Cells")
    
    return ax  # Return the axis object
    
def highlight_rhombus(rhombus_vertices, ax=None, color='green', alpha=0.5):
    """
    Highlight the overlapping rhombus area on a map.
    
    Parameters:
    - rhombus_vertices: List of rhombus vertices [(lat1, lon1), (lat2, lon2), ...].
    - ax: Matplotlib axis object (optional, for overlaying on existing plots).
    - color: Color of the fill for the rhombus.
    - alpha: Transparency of the fill.
    """
    rhombus_vertices.append(rhombus_vertices[0])  # Close the rhombus
    lats, lons = zip(*rhombus_vertices)  # Separate latitudes and longitudes
    
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
    
    # Fill the rhombus with the specified color and transparency
    ax.fill(lons, lats, color=color, alpha=alpha, label="Overlapping Area")
    
    # Add labels and grid for clarity
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid()
    return ax


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
    satellite_EIRP_total = 10*np.log10((10**(satellite_EIRP_density/10))*max_Bandwidth_per_beam)
    A_z = 1*10**(-1) #Based on the figure 4 of the document ITU-R P.676-13 other values for 30 GHz is 0.2 and 5 GHz is 0.04 in dB
    G_max_Rx = 4 #dBi    
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

    base_lat, base_lon = 53.110987, 8.851239  
    base_station_point = wgs84.latlon(base_lat, base_lon)
    radius_km = 25

    # Calculate hexagonal vertices
    vertices = hexagon_vertices(base_lat, base_lon, radius_km)
    # get the sorted vertices to calculate the two edges
    vertices_sorted_in_lat_lon  = vertices_sorting(vertices)
    # Southern side points of the second hexagon
    lat1, lon1 = vertices_sorted_in_lat_lon[1]
    lat2, lon2 = vertices_sorted_in_lat_lon[2]
    
    # Radius of the new hexagon
    radius_new_hexagon = 25  # km
    # Compute the center of the new hexagon
    center_lat, center_lon = compute_new_hexagon_center(lat1, lon1, lat2, lon2, 
                                                        radius_new_hexagon)
    # Compute vertices of the new hexagon
    new_hexagon_vertices = hexagon_vertices(center_lat, center_lon, radius_new_hexagon)
    new_hexagon_vertices_sorted = vertices_sorting(new_hexagon_vertices)
    # Get northern edge point of basestation
    north_lat, north_lon = vertices_sorted_in_lat_lon[0]
    # Get satellite southern point 
    south_lat, south_lon = new_hexagon_vertices_sorted[-1]

    # Define the needed points in wgs84 format:
    base_station_point = wgs84.latlon(base_lat, base_lon)
    bs_northern_edge_point = wgs84.latlon(north_lat, north_lon)
    satellite_beam_center = wgs84.latlon(center_lat, center_lon)
    satellite_souther_edge_point = wgs84.latlon(south_lat, south_lon)
    groundstation = satellite_beam_center
    
    noise_figure_db = 7 # in dB
    temperature_k = 25 + 273.15 # in Kelvin
    GS_distance_from_Earth_center = LA.norm(satellite_beam_center.itrs_xyz.km)
    DF = LG.simulateConstellation(LEOs, satellite_beam_center, 20, time1, time2, ts = None, safetyMargin = 0)
    DF2 = DF.reset_index()
    # create an empty DataFrame to store satellite position data
    sat_position_df = pd.DataFrame(columns=['Satellite ID', 'Time', 'Sat Position (km)', 'GS Position (km)', 'Distance from Earth Surface (km)', 'distance to GS (km)', 'Elevation Angle (degree)'])


    plot_hexagon(vertices, (base_lat, base_lon))
    # Get the overlapping rhombus vertices
    rhombus_vertices = overlapping_rhombus((north_lat, north_lon),
                                           (south_lat, south_lon),
                                           vertices_sorted_in_lat_lon
    )
    rhombus_vertices_sorted = sort_vertices_counterclockwise(rhombus_vertices)
    # Print or plot the rhombus
    print("Rhombus vertices:", rhombus_vertices_sorted)
    # Plot hexagons and get the axis
    ax = plot_two_hexagons(vertices, (base_lat, base_lon), new_hexagon_vertices, (center_lat, center_lon))
    # Highlight the overlapping rhombus on the same plot
    highlight_rhombus(rhombus_vertices_sorted, ax=ax, color='green', alpha=0.5)
    # Show the final plot
    plt.grid()
    plt.show()
    
    sat_arr = []
    visiting_start_GS_LEO = []
    visiting_end_GS_LEO = []
    distance_GS_LEOs = []
    
    generation_intervals = 10 # seconds.
    number_samples = generation_intervals * 1000 #miliseconds; for microseconds multiply by 10e6
    satellite_fading_channel = FadingSimulation(num_samples = number_samples, fs = 1000, K = 10, N = 64, h = h_LEO, Doppler_compensate = 'Yes') 
    Shadowing = ShadowingFading(tau = 1, N = number_samples)
    #%% All Data Frames initializations except satellite positions:
        # satellite to Ground station received power data frame
    Satellite_P_Rx_data_set = pd.DataFrame(columns=['Satellite ID', 'Time', 'Elevation Angle (degree)','Distance (km)', 'LoS Prob (%)', 'P_Rx_fspl_ShF_SSF (dBW)', 'SNR (dB)'])
    #%%#  *** The main loop of the program to generate position and channel for each involving element; The reference is based on the satellite movements *** #############
     
   ########################### First loop over satellites based on their rise over the area ############################
         # the 45th emerging satellite is equal to the 1st emerging satellite, we consider up to then it is not possible to change the frequency band
     #for i in range(45):
         # for in whole of the constellation in 24 hours
     #for i in range(len(DF2)):
         # for over 4 satellites in a group
     #for i in range(3,7):
    for i in range(83,84):
         Sat_ID = DF2.iloc[i,0]
         Sat_ID_int = int(Sat_ID[3:])
         t_rise  = DF2.iloc[i,1]
         t_set = DF2.iloc[i,2]
         t_rise_now = ts.utc(t_rise.year, t_rise.month, t_rise.day, t_rise.hour, t_rise.minute, t_rise.second)
         t_set_now = ts.utc(t_set.year, t_set.month, t_set.day, t_set.hour, t_set.minute, t_set.second)
         visibility_sec = LG.difference_time_in_seconds(t_rise_now, t_set_now)
         # In oder to calculate in miliseconds you need to convert visibility_sec to miliseconds by: visibility_sec*1000 and then in 
         # datetime.timedelta(seconds=i1) write instead of seconds as microseconds = i1*1000 because it doesn't accept miliseconds 
    
        ########################### Second loop over visibility duration of each satellite ############################
         
         for i1 in range(0,visibility_sec*1000):

             time_now = t_rise_now+datetime.timedelta(microseconds=i1*1000)
             
             position_LEO = LEOs[Sat_ID_int-1].at(time_now)
             
             position_GS = groundstation.at(time_now).position.km
             
             # calculate the position of the southern point of satellite beam
             position_bs_northern_edge = bs_northern_edge_point.at(time_now).position.km
             distance_beam_center_to_edge_point = LG.distance(position_GS, position_bs_northern_edge)
             
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
             
             # calculate the Shadowed-Rician fading channel temporally correlated as:
                 # we generate the channel for each generation_intervals seconds since the change of elevation angle each generation_intervals seconds is significant
                 # we generate per each second 1000 samples means TTi is 1 ms (10e6 samples if TTi is in microseconds)
                 # Here since TTI is 1 ms, we feed at each i2 one sample out of generated channels 
             if i1 % generation_intervals == 0:
                 print(f"Generating samples at i={i}, i1={i1}")
                 shadowed_rician_channel_samples_interval = satellite_fading_channel.run_simulation(elevation_angle, f, distance_GS_sat)
                 shadowing_samples_interval = Shadowing.SF_LOS_calc(elevation_angle, 'LOS', 'SBand')
                 
                     
             start_idx = i1 % generation_intervals
             end_idx = start_idx + 1
        
             if end_idx <= len(shadowed_rician_channel_samples_interval):
                Satellite_Channel_fading_samples = shadowed_rician_channel_samples_interval[start_idx:end_idx]
                Satellite_Shadowing_samples = shadowing_samples_interval[start_idx:end_idx]
             else:
                print(f"Index out of range at i={i}, i1={i1}")
             
             # calculate only the fspl as function of distance and frequecny
             fspl_solo = Rx_power().FSPl_only(f, distance_GS_sat)
             
             atmospheric_loss = Rx_power().atmospheric_att(A_z, elevation_angle)
             
             #Satellite_Shadowing_samples, Satellite_Shadowing_samples_nlos, clutter_loss = Rx_power().SF_LOS_calc(elevation_angle, 'LOS', 'SBand')
             
             #if lOS_prob > 50:            
                 #P_received_fspl_ShF_SSF = (satellite_EIRP_total + G_max_Rx + 20*np.log10(np.abs(Satellite_Channel_fading_samples))) - fspl_solo - atmospheric_loss - Satellite_Shadowing_samples
             #else:
                 #P_received_fspl_ShF_SSF = (satellite_EIRP_total + G_max_Rx + 20*np.log10(np.abs(Satellite_Channel_fading_samples))) - fspl_solo - atmospheric_loss - Satellite_Shadowing_samples_nlos - clutter_loss
             
             P_received_fspl_ShF_SSF = (satellite_EIRP_total + G_max_Rx + 20*np.log10(np.abs(Satellite_Channel_fading_samples))) - fspl_solo - atmospheric_loss -Satellite_Shadowing_samples
             P_recieved_at_edge_point = P_received_fspl_ShF_SSF + 10 * 2 * np.log10(1 - distance_beam_center_to_edge_point/satellite_beam_diameter/2)
             Noise_power = Rx_power().Noise_power_with_NoiseFigure(noise_figure_db, max_Bandwidth_per_beam, temperature_k)              
             
             new_row_P_Rx_data_set = pd.DataFrame({
                 'Satellite ID': [Sat_ID],
                 'Time': [time_now.utc_datetime()],
                 'Elevation Angle (degree)': [elevation_angle],
                 'Distance (km)': [distance_GS_sat],
                 'LoS Prob (%)': [lOS_prob],
                 'P_Rx_fspl_ShF_SSF (dBW)': [P_received_fspl_ShF_SSF],
                 'SNR (dB)': [P_received_fspl_ShF_SSF - Noise_power]
                 })
             Satellite_P_Rx_data_set = pd.concat([Satellite_P_Rx_data_set, new_row_P_Rx_data_set], ignore_index=True)
             print(i1)
         print(i)
    
