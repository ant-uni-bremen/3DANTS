#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:23:37 2025

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
    f = 2.0e9;
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
    plt.figure(figsize=(18, 12), dpi = 1000)
    plt.bar(total_visibility_minutes.index, total_visibility_minutes.values)
    plt.xlabel('Satellite Name')
    plt.ylabel('Total Visibility Duration (minutes)')
    plt.title('Satellite Visibility Durations')
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.tight_layout()
    plt.show()
    #%% time series analysis 
    # create an empty DataFrame to store satellite position data
    sat_position_df = pd.DataFrame(columns=['Satellite ID', 'Time', 'Sat Position (km)', 'GS Position (km)', 'Distance from Earth Surface (km)', 'distance to GS (km)', 'Elevation Angle (degree)', 'satellite velocity (km)', 'v_LOS (kmps)'])
    sat_orbital_parameters_df = pd.DataFrame(columns=['Satellite ID', 'Time', 'Theta_el_sat_see_vsat', 'Theta_az_sat_see_vsat', 'theta_el_vsat_see_sat', 'theta_az_vsat_see_sat'])

    sat_arr = []
    visiting_start_GS_LEO = []
    visiting_end_GS_LEO = []
    distance_GS_LEOs = []
    for i in range(0,len(DF2)):
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
            """
            r_sat_vsat_global = position_LEO.xyz.km - groundstation.at(time_now).position.km
            # Step 2: Compute space angles
            r_sat_vsat_global_norm = np.linalg.norm(r_sat_vsat_global)
            r_x_global, r_y_global, r_z_global = r_sat_vsat_global
            
            # Elevation angle
            theta_el_vsat_see_sat = np.arcsin(r_z_global / r_sat_vsat_global_norm)
            
            # Azimuth angle
            theta_az_vsat_see_sat = np.arctan2(r_y_global, r_x_global)
            """
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



#%% Group by 'Time' and count unique satellites
visible_sats_per_time = sat_position_df.groupby('Time')['Satellite ID'].nunique()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(visible_sats_per_time.index, visible_sats_per_time.values, label='Visible Satellites')
plt.xlabel('Time')
plt.ylabel('Number of Visible Satellites')
plt.title('Number of Visible Satellites Over Time')
plt.grid(True)
plt.legend()
plt.show()
#%% Group by 'Time' and get max elevation angle
max_elevation_per_time = sat_position_df.groupby('Time')['Elevation Angle (degree)'].max()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(max_elevation_per_time.index, max_elevation_per_time.values, color='orange', label='Max Elevation')
plt.xlabel('Time')
plt.ylabel('Maximum Elevation Angle (degrees)')
plt.title('Maximum Elevation Angle Over Time')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6), dpi = 1024)
plt.hist(sat_position_df['Elevation Angle (degree)'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Elevation Angle (degrees)')
plt.ylabel('Frequency (seconds)')
plt.title('Distribution of Elevation Angles')
plt.grid(True)
plt.show()
#%% Scatter Plot of v_LOS vs. Elevation Angle
plt.figure(figsize=(8, 6))
plt.scatter(sat_position_df['Elevation Angle (degree)'], sat_position_df['v_LOS (kmps)'], s=5, alpha=0.5, color='purple')
plt.xlabel('Elevation Angle (degrees)')
plt.ylabel('v_LOS (km/s)')
plt.title('v_LOS vs. Elevation Angle')
plt.grid(True)
plt.show()
#%% Elevation Angle Over Time for Each Satellite Pass
plt.figure(figsize=(12, 6))
for sat_id in sat_position_df['Satellite ID'].unique()[:5]:  # Limit to first 5 satellites
    sat_data = sat_position_df[sat_position_df['Satellite ID'] == sat_id]
    plt.plot(sat_data['Time'], sat_data['Elevation Angle (degree)'], label=sat_id)
plt.xlabel('Time')
plt.ylabel('Elevation Angle (degrees)')
plt.title('Elevation Angle Over Time per Satellite')
plt.legend()
plt.grid(True)
plt.show()
#%% Doppler analysis 

v_LOS_df  = sat_position_df['v_LOS (kmps)']
Doppler_frequency_shift_with_sign = ((v_LOS_df * 1000)/(3.0e8)) * f
plt.figure(figsize=(8, 6), dpi = 1024)
plt.hist(Doppler_frequency_shift_with_sign, bins=100)
plt.xlabel("Doppler shift (Hz)")
plt.ylabel("Count")
plt.title("Signed Doppler Shift Distribution")
plt.show()

plt.figure(figsize=(8, 6), dpi = 1024)
plt.hist(np.abs(Doppler_frequency_shift_with_sign), bins=100)
plt.xlabel("Absolute Doppler shift (Hz)")
plt.ylabel("Count")
plt.title("Magnitude of Doppler Shift")
plt.show()
#%% Ground Station’s Perspective: Polar Plot of Satellite Positions: 1 time stamp

# Example data: Replace with your DataFrame
# sat_position_df = your_data_frame
specific_time = sat_orbital_parameters_df['Time'].unique()[45000]  # First timestamp
data_at_time = sat_orbital_parameters_df[sat_orbital_parameters_df['Time'] == specific_time]

# Create polar plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')

# Plot each satellite
for _, row in data_at_time.iterrows():
    az = np.rad2deg(row['theta_az_vsat_see_sat'])
    el = np.rad2deg(row['theta_el_vsat_see_sat'])
    ax.scatter(az, el, label=row['Satellite ID'])

# Customize plot
ax.set_ylim(0, 90)  # Elevation from 0° to 90°
ax.set_theta_zero_location('N')  # 0° at North
ax.set_theta_direction(-1)  # Clockwise
plt.title(f'Satellite Positions at {specific_time}')
plt.legend()
plt.show()
#%% Ground Station’s Perspective: Polar Plot of Satellite Positions: multiple time stamp
# Get unique timestamps and select every xth for example here 1000
unique_times = sat_orbital_parameters_df['Time'].unique()
selected_times = unique_times[::1000]  # Adjust the step (10) based on your needs
# Create a 2x2 grid of polar plots
fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': 'polar'})
axs = axs.flatten()  # Flatten the 2D array for easier indexing

for i, time in enumerate(selected_times[:4]):  # Limit to first 4 timestamps
    data_at_time = sat_orbital_parameters_df[sat_orbital_parameters_df['Time'] == time]
    
    for _, row in data_at_time.iterrows():
        az = np.rad2deg(row['theta_az_vsat_see_sat'])
        el = np.rad2deg(row['theta_el_vsat_see_sat'])
        axs[i].scatter(az, el, label=row['Satellite ID'])
    
    # Customize each subplot
    axs[i].set_ylim(0, 90)
    axs[i].set_theta_zero_location('N')
    axs[i].set_theta_direction(-1)
    axs[i].set_title(f'Time: {time}')
    
    if len(data_at_time) <= 5:  # Adjust threshold for legend
        axs[i].legend()

plt.tight_layout()
plt.show()
#%% SATELLITE’s Perspective: Polar Plot of groundstation Positions: multiple time stamp of satellite passage 
# Select a satellite
sat_id = 'Sat 50'  # Replace with your satellite ID
sat_data = sat_orbital_parameters_df[sat_orbital_parameters_df['Satellite ID'] == sat_id]

# Create polar plot
fig = plt.figure(figsize=(8, 8), dpi = 1024)
ax = fig.add_subplot(111, projection='polar')

# Plot ground station’s track
az = np.rad2deg(sat_data['Theta_az_sat_see_vsat'])
el = np.rad2deg(sat_data['Theta_el_sat_see_vsat'])
ax.scatter(az, el, label='Ground Station Track')

ax.set_ylim(0, 90)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
plt.title(f'Ground Station from {sat_id} Perspective')
plt.legend()
plt.show()

#%%

# Define time step for trajectory (10 seconds)
delta_t = pd.Timedelta(seconds=10)

# Get unique timestamps and select four for subplots
unique_times = sat_orbital_parameters_df['Time'].unique()
num_times = len(unique_times)
step = num_times // 10
selected_times = unique_times[::step][:4]

# Assign unique colors to all satellites appearing in selected times
all_sats = sat_orbital_parameters_df[sat_orbital_parameters_df['Time'].isin(selected_times)]['Satellite ID'].unique()
colors = {sat_id: plt.cm.get_cmap('tab10')(i % 10) for i, sat_id in enumerate(all_sats)}

# Create 2x2 grid of polar subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': 'polar'}, dpi = 1024)
axs = axs.flatten()

for i, t in enumerate(selected_times):
    ax = axs[i]
    
    # Define the 5 timestamps for trajectories: t-2δ, t-δ, t, t+δ, t+2δ
    time_points = [
        t - 2 * delta_t,
        t - delta_t,
        t,
        t + delta_t,
        t + 2 * delta_t
    ]
    
    # Filter data for satellites visible at time t
    data_at_t = sat_orbital_parameters_df[sat_orbital_parameters_df['Time'] == t]
    visible_sats = data_at_t['Satellite ID'].unique()
    
    # Plot trajectories for each visible satellite
    for sat_id in visible_sats:
        # Get all data for this satellite
        sat_data = sat_orbital_parameters_df[sat_orbital_parameters_df['Satellite ID'] == sat_id]
        # Filter positions at the 5 time points where available
        traj_data = sat_data[sat_data['Time'].isin(time_points)]
        if not traj_data.empty:
            traj_data = traj_data.sort_values('Time')  # Ensure chronological order
            az = np.rad2deg(traj_data['theta_az_vsat_see_sat'])
            el = np.rad2deg(traj_data['theta_el_vsat_see_sat'])
            color = colors[sat_id]
            # Plot trajectory as a line
            ax.plot(az, el, color=color, alpha=0.5, linewidth=1.5)
    
    # Overlay positions at exactly time t
    for _, row in data_at_t.iterrows():
        az = np.radians(row['theta_az_vsat_see_sat'])
        el = row['theta_el_vsat_see_sat']
        color = colors[row['Satellite ID']]
        ax.scatter(az, el, color=color, s=100, marker='o', zorder=5)
    
    # Customize plot
    ax.set_ylim(0, 90)  # Elevation from 0° to 90°
    ax.set_theta_zero_location('N')  # North at top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_title(f'Time: {t.strftime("%Y-%m-%d %H:%M:%S")}', pad=20)

# Adjust layout and display
plt.tight_layout()
plt.savefig('satellite_trajectories.png')
#%% 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import datetime

# Create a 2x2 grid of polar plots
fig, axs = plt.subplots(2, 2, figsize=(20, 20), subplot_kw={'projection': 'polar'}, dpi = 1024)
axs = axs.flatten()  # Flatten the 2D array for easier indexing

# Define number of consecutive positions to show
track_length = 50
time_step = 90  # seconds between consecutive positions

# Get all unique timestamps in the DataFrame
all_times = sorted(sat_orbital_parameters_df['Time'].unique())

# Select 4 reference timestamps evenly distributed across the data
if len(all_times) >= 4:
    selected_times = [all_times[i] for i in np.linspace(0, len(all_times)-1, 4, dtype=int)]
else:
    selected_times = all_times  # Use all available times if less than 4

# Color map for different satellites
cmap = plt.cm.get_cmap('tab10')
satellite_colors = {}

# Process 4 selected timestamps
for i, center_time in enumerate(selected_times[:4]):
    center_time_idx = np.where(unique_times == center_time)[0][0]
    
    # Get unique satellite IDs at this timestamp
    data_at_time = sat_orbital_parameters_df[sat_orbital_parameters_df['Time'] == center_time]
    satellites = data_at_time['Satellite ID'].unique()
    
    # Assign colors to satellites if not already assigned
    for j, sat_id in enumerate(satellites):
        if sat_id not in satellite_colors:
            satellite_colors[sat_id] = cmap(j % 10)
    
    # For each satellite, plot track of positions
    for sat_id in satellites:
        # Find positions for this satellite around the center time
        track_positions = []
        
        # Calculate actual datetime object from the center_time (which might be a string or timestamp)
        if isinstance(center_time, str):
            center_datetime = pd.to_datetime(center_time)
        else:
            center_datetime = center_time
            
        # Look back and forward from center time using the specified time_step
        half_length = track_length // 2
        for offset in range(-half_length, half_length + 1):
            try:
                # Calculate the target time by adding/subtracting seconds
                time_offset = offset * time_step  # Convert to seconds
                target_datetime = center_datetime + pd.Timedelta(seconds=time_offset)
                
                # Find the closest available timestamp in our data
                # This is important because the exact time might not exist in the data
                closest_time_idx = np.argmin([abs((t - target_datetime).total_seconds()) 
                                             for t in pd.to_datetime(all_times)])
                target_time = all_times[closest_time_idx]
                
                # Find the satellite data for this specific time
                sat_data = sat_orbital_parameters_df[
                    (sat_orbital_parameters_df['Time'] == target_time) & 
                    (sat_orbital_parameters_df['Satellite ID'] == sat_id)
                ]
                
                if not sat_data.empty:
                    # Get az/el and add to track
                    az = np.rad2deg(sat_data['theta_az_vsat_see_sat'].values[0])
                    el = np.rad2deg(sat_data['theta_el_vsat_see_sat'].values[0])
                    track_positions.append((az, el))
            except Exception as e:
                print(f"Error processing offset {offset} for {sat_id}: {e}")
        
        # Plot track if we have positions
        if track_positions:
            # Convert to arrays for plotting
            az_values, el_values = zip(*track_positions)
            
            # Convert elevation to radius (90-el so zenith is at center)
            radius_values = [90-el for el in el_values]
            
            # Plot satellite track
            axs[i].plot(np.deg2rad(az_values), radius_values, '-', 
                      color=satellite_colors[sat_id], alpha=0.7, linewidth=1)
            
            # Plot points with decreasing size to show direction
            for j, (az, r) in enumerate(zip(np.deg2rad(az_values), radius_values)):
                size = 20 * (j+1)/len(radius_values)  # Size increases with recency
                axs[i].scatter(az, r, s=size, color=satellite_colors[sat_id], 
                             alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # Add satellite ID text near the most recent (largest) point
            last_idx = len(az_values) - 1
            if last_idx >= 0:
                axs[i].annotate(sat_id, 
                              xy=(np.deg2rad(az_values[last_idx]), radius_values[last_idx]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, color=satellite_colors[sat_id])
    
    # Customize each subplot
    axs[i].set_ylim(0, 90)  # Radius from 0 (zenith) to 90 (horizon)
    axs[i].set_theta_zero_location('N')  # North at top
    axs[i].set_theta_direction(-1)  # Clockwise
    axs[i].set_rticks([0, 30, 60, 90])  # Elevation circles at 90°, 60°, 30°, 0°
    axs[i].set_yticklabels(['90°', '60°', '30°', '0°'])  # Label as elevation angles
    
    # Add elevation circles and labels
    for angle in [30, 60]:
        axs[i].text(np.deg2rad(45), 90-angle, f"{angle}°", 
                  fontsize=8, ha='center', va='center', alpha=0.7)
    
    # Add azimuth labels
    for az in [0, 90, 180, 270]:
        axs[i].text(np.deg2rad(az), 95, ['N', 'E', 'S', 'W'][az//90], 
                  fontsize=10, ha='center', va='center')
    
    # Add timestamp title
    time_str = pd.to_datetime(center_time).strftime('%Y-%m-%d %H:%M:%S')
    axs[i].set_title(f'Time: {time_str}', fontsize=12)
    
    # Add subtitle showing the time step
    axs[i].text(0, 0, f"Track interval: {time_step} seconds", 
              fontsize=8, ha='center', va='center', 
              transform=axs[i].transAxes)

# Add a common title
plt.suptitle('Satellite Tracks from Ground Station Perspective\n(5 consecutive positions per satellite)', 
             fontsize=16, y=0.98)

# Add explanation for track direction
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markersize=5, label='First Position'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markersize=10, label='Last Position')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
           bbox_to_anchor=(0.5, 0.02), fontsize=10)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08)
plt.show()