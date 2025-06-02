#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:41:40 2023

@author: vakilifard
"""
from skyfield.api import load, wgs84
import numpy as np
import math
from scipy.stats import rice

class terresterial_network:
    np.random.seed(42)
    def __init__(self, fc, environment):
        self.fc = fc/1e9 # fc is in HZ but self.fc is in GHz
        self.environment = environment
        self.c = 3.0e8
################################################# UE Generation #################################################        

    def UE_random_PPP_generation(self, center, radius ,num_points):
        # Generate random angles and radii for each point
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        radii = np.sqrt(np.random.uniform(0, radius**2, num_points))
        
        # Convert polar coordinates to Cartesian coordinates
        x_coords = center[0] + radii * np.cos(angles)
        y_coords = center[1] + radii * np.sin(angles)
        z_cord  = center[2] - 0.034
        z_coords = np.full_like(x_coords, z_cord)
        # Combine x and y coordinates into a single array of positions
        positions = np.column_stack((x_coords, y_coords, z_coords))       
        distance = np.linalg.norm(positions - np.full_like(positions,center), axis=1)
        pathloss_NLOS = 36.7 * np.log10(distance*1000) + 20.5 * np.log10(self.fc) + 21.4
        return positions    

  # The following function clusters the UEs based on their distance to the BaseStations      
    def generate_ue_clusters(self, base_station_positions, num_ues_per_bs, radius):
        # Initialize lists to store clustered UEs
        UE_cluster_per_BS = [[] for _ in range(len(base_station_positions))]
    
        # Generate UE positions around each base station
        for i, bs_position in enumerate(base_station_positions):
            ue_positions = self.UE_random_PPP_generation(bs_position, radius, num_ues_per_bs)
            UE_cluster_per_BS[i].extend(ue_positions)
    
        return UE_cluster_per_BS
    
    
    
    def UAV_PPP_generation_inisde_a_cone(self, center, r, h ,numbPoints):
        # Generate random angles and radii for each point
        zz = h * (np.random.rand(numbPoints, 1))**(1/3)  # z coordinates
        theta = 2 * np.pi * (np.random.rand(numbPoints, 1))  # angular coordinates
        rho = r * (zz / h) * np.sqrt(np.random.rand(numbPoints, 1))  # radial coordinates

        # Convert from polar to Cartesian coordinates
        xx = rho * np.cos(theta)
        yy = rho * np.sin(theta)

        # Shift tip of cone to (xx0, yy0, zz0)
        xx += center[0]
        yy += center[1]
        zz += center[2]
        positions = np.column_stack((xx, yy, zz))
        return positions    

  # The following function clusters the UEs based on their distance to the BaseStations      
    def generate_uavs_clusters(self, base_station_positions, num_uavs_per_bs, radius_of_uavs_cluster, max_height):
        # Initialize lists to store clustered UEs
        UAVs_cluster_per_BS = [[] for _ in range(len(base_station_positions))]
    
        # Generate UE positions around each base station
        for i, bs_position in enumerate(base_station_positions):
            uavs_position = self.UAV_PPP_generation_inisde_a_cone(bs_position, radius_of_uavs_cluster, max_height, num_uavs_per_bs)
            UAVs_cluster_per_BS[i].extend(uavs_position)
    
        return UAVs_cluster_per_BS
    
    
    
    
    
    
    def UEs_cartesian_to_latlon(self, coordinates):
        # Define the radius of the Earth (assuming a spherical Earth model).
        # Initialize an empty array to store the latlon coordinates.
        latlon_coordinates = np.zeros((coordinates.shape[0], coordinates.shape[1], 2))

        for i in range(coordinates.shape[0]):  # Loop over base stations
            for j in range(coordinates.shape[1]):  # Loop over users
                x, y, z = coordinates[i, j, :]

                # Calculate the longitude (lon) in degrees.
                lon = math.degrees(math.atan2(y, x))

                # Calculate the latitude (lat) in degrees.
                lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))

                latlon_coordinates[i, j, :] = np.array([lat, lon])

        return latlon_coordinates
    
    def skyfield_position_for_UEs(self, latlon_array):
        # Initialize an empty list to store the wgs84.latlon objects
        positions = []

        # Iterate through the array and create wgs84.latlon objects
        for i in range(latlon_array.shape[0]):  # Loop over base stations
            #base_station_positions = []
            for j in range(latlon_array.shape[1]):  # Loop over UEs
                lat, lon = latlon_array[i, j, :]
                position = wgs84.latlon(lat, lon)
                #base_station_positions.append(position)
                #positions.append(base_station_positions)
                positions.append(position)

        return positions
    
    def get_UEs_positions_at_time_now(self, time_now, base_station_skyfield_list, height_UE):
        base_station_position_xyz = []
        for base_station in base_station_skyfield_list:
            position_km = base_station.at(time_now).position.km + np.array([0,0,height_UE/1000])
            base_station_position_xyz.append(position_km)
        return base_station_position_xyz
    
    
    def UEs_random_movement_inside_BS_area(self, UEs_array_positions_2D, radius_of_movement):
        points_number = 1
        # Generate random angles and radii for each point
        UEs_array_positions_2D_new = np.zeros(UEs_array_positions_2D.shape)
        for i in range(UEs_array_positions_2D.shape[0]):
            for j in range(UEs_array_positions_2D.shape[1]):
                angles = np.random.uniform(0, 2 * np.pi, points_number)
                radii = np.sqrt(np.random.uniform(0, radius_of_movement**2, points_number))
                # Convert polar coordinates to Cartesian coordinates
                points_x = UEs_array_positions_2D[i,j,0] + radii * np.cos(angles)
                points_y = UEs_array_positions_2D[i,j,1] + radii * np.sin(angles)
                UEs_array_positions_2D_new[i,j,0] = points_x
                UEs_array_positions_2D_new[i,j,1] = points_y
        return UEs_array_positions_2D_new
#########################################################3 Base Station generation #########################################
    
    def generate_base_station_positions(self,
                                    center_x: float,
                                    center_y: float,
                                    circle_radius: float,
                                    num_base_stations: int,
                                    fixed_z: float,
                                    r_bs: float,
                                    max_trials_per_bs: int = 1000):
        """
        Generate `num_base_stations` points uniformly in a circle of radius
        `circle_radius` around (center_x, center_y), such that each pair of
        points is at least 2*r_bs apart (no coverage‐circle overlaps).
        """
    
        base_stations = []
    
        trials = 0
        # keep trying until we have enough stations or exceed total trial budget
        while len(base_stations) < num_base_stations:
            if trials > num_base_stations * max_trials_per_bs:
                raise RuntimeError(f"Could not place {num_base_stations} non‐overlapping stations "
                                   f"after {trials} attempts; try reducing density or increasing "
                                   f"circle_radius.")
            trials += 1
    
            # sample uniformly in circle
            r = math.sqrt(np.random.rand()) * circle_radius
            theta = 2 * math.pi * np.random.rand()
            x = center_x + r * math.cos(theta)
            y = center_y + r * math.sin(theta)
    
            # check separation against all existing stations
            too_close = False
            for (xi, yi, _) in base_stations:
                if math.hypot(x - xi, y - yi) < 2 * r_bs:
                    too_close = True
                    break
    
            if not too_close:
                base_stations.append((x, y, fixed_z))
    
        return base_stations
    
    
    def cartesian_to_latlon(self, coordinates):
        # Define the radius of the Earth (assuming a spherical Earth model).    
        # Initialize an empty list to store the latlon coordinates.
        latlon_coordinates = []
    
        for xyz in coordinates:
            x, y, z = xyz
    
            # Calculate the longitude (lon) in degrees.
            lon = math.degrees(math.atan2(y, x))
    
            # Calculate the latitude (lat) in degrees.
            lat = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
    
            latlon_coordinates.append((lat, lon))
    
        return latlon_coordinates
    
    def skyfield_position_for_BaseStations(self, BaseStations_lat_lon):
        # Initialize an empty list to store the wgs84.latlon objects
        base_station_positions = []
        # Iterate through the array and create wgs84.latlon objects
        for lat, lon in BaseStations_lat_lon:
            position = wgs84.latlon(lat, lon)
            base_station_positions.append(position)
        return base_station_positions
    
    def get_base_station_positions_at_time_now(self, time_now, base_station_skyfield_list, height_BS):
        base_station_position_xyz = []
        for base_station in base_station_skyfield_list:
            position_km = base_station.at(time_now).position.km + np.array([0,0,0.035])
            base_station_position_xyz.append(position_km)
        return base_station_position_xyz

            
########################################################## Communication Channel ##############################        

    def Pr_LOS_RMa(self, d_2D_out, h_UT):
        Pr_LOS = 0  # Initialize the probability of LOS to 0
    
        if self.environment == 'Suburban':
            if d_2D_out > 10:
                Pr_LOS = np.exp(-(d_2D_out - 10) / 1000)
            else:
                Pr_LOS = 1
        elif self.environment == 'Urban':
            if d_2D_out > 10:
                Pr_LOS = 18 / d_2D_out + np.exp(-d_2D_out / 36) * (1 - 18 / d_2D_out)
            else:
                Pr_LOS = 1
        elif self.environment == 'DenseUrban':
            C_h_UT = 0
            if 13 < h_UT <= 23:
                C_h_UT = math.pow((h_UT - 13) / 100, 1.5)
    
            if d_2D_out > 10:
                Pr_LOS = (18 / d_2D_out + np.exp(-d_2D_out / 63) * (1 - 18 / d_2D_out)) * (
                        1 + 1.25 * C_h_UT * math.pow(d_2D_out / 100, 3) * np.exp(-d_2D_out / 150))
            else:
                Pr_LOS = 1
    
        return Pr_LOS

    
    def pathloss_calculator_up_to_7GHz(self, GS_position, uav_position, h_BS, h_UT):
        # This pathloss is calculated based on the 3GPP TR 38.901 page 27 which is valid only if the altitude of the object is up to 35 meter. It is reported for RMa (Rural Macro), UMa (Urban Macro) and UMi (Urban Micro) scenario
        # Based on 3gpp claim fc must be in Hz, h_Bs and h_UT and all other parametrs related to distnace be in meter
        
        d_BP = 2*np.pi*h_BS*h_UT*(self.fc*1e9)/self.c
        d3D = GS_position-uav_position
        distance = np.linalg.norm(d3D) * 1000 #convert it to meter
        d2D = GS_position[:2] - uav_position[:2]
        d_2D = np.linalg.norm(d2D) * 1000
        prob_los = self.Pr_LOS_RMa(d_2D, h_UT)
        PL_RMa_LOS = 0.0
        PL_RMa_NLOS = 0.0
        if self.environment == 'Suburban':
            h = 5
            W = 20
            if prob_los >= 0.5:
                PL_1 = 20 * math.log10(40 * math.pi * distance * self.fc  / 3) + min(0.03 * math.pow(h, 1.72), 10) * math.log10(distance) - min(0.044 * math.pow(h, 1.72), 14.77) + 0.002 * math.log10(h) * distance
                PL_2 = PL_1 + 40 * math.log10(distance / d_BP)

                if 10 < d_2D and d_2D <= d_BP:
                    PL_RMa_LOS = PL_1 + np.random.normal(0, 4, 1)
                else:
                    PL_RMa_LOS = PL_2 + np.random.normal(0, 6, 1)
            else:
                PL_RMa_NLOS_2 = 161.04 - 7.1 * math.log10(40 * W) + 7.5 * math.log10(h) - \
                                (24.37 - 3.7 * (h / h_BS) * (h / h_BS)) * math.log10(h_BS) + \
                                (43.42 - 3.1 * math.log10(h_BS)) * (math.log10(distance) - 3) + \
                                20 * math.log10(self.fc) - (3.2 * math.pow(math.log10(11.75 * h_UT), 2) - 4.97)
                PL_RMa_NLOS = PL_RMa_NLOS_2 + np.random.normal(0, 8, 1)
                
        return PL_RMa_LOS, PL_RMa_NLOS
        
        """    
        # previous code on the main code page is as: 
        PL_Shad_LOS = [[] for _ in range(len(np.array([BaseStation_positions_time_now])))]
        PL_Shad_NLOS = [[] for _ in range(len(np.array([BaseStation_positions_time_now])))]
        for bs in range(np.array([UE_positions_served_by_BSs]).shape[1]): # A for loop over BaseStations then in the next line we feed the function with BS1 and UEs positions of BS1: BS1 position: 1*3, UEs position: 16*3
            PL_Sha_LOS, PL_Sha_NLOS = Terresterial.pathloss_BaseStations_to_UEs(np.array(BaseStation_positions_time_now[bs]), np.array(UE_positions_served_by_BSs[bs]), height_BS, 1.2)
            PL_Shad_LOS[bs].extend(PL_Sha_LOS)
            PL_Shad_NLOS[bs].extend(PL_Sha_NLOS)
        aa = np.array([PL_Shad_LOS])
        bb = np.array([PL_Shad_NLOS])
        """
    def pathloss_BaseStations_to_UEs(self, BSs_position, UEs_positions, UEs_Shadowing_per_BS_from_Gaussian_field,h_BS, h_UT):
        PL_Shad_LOS = [[] for _ in range(len(BSs_position))]
        PL_Shad_NLOS = [[] for _ in range(len(BSs_position))]
        for bs in range(UEs_positions.shape[0]): # A for loop over BaseStations then in the next line we feed the function with BS1 and UEs positions of BS1: BS1 position: 1*3, UEs position: (number of UEs)*3
            ue_positions = UEs_positions[bs]
            BS_position = BSs_position[bs]
            PL_Sha_LOS = np.zeros_like(ue_positions[:,0])
            PL_Sha_NLOS = np.zeros_like(ue_positions[:,0])
            d_BP = 2 * np.pi * h_BS * h_UT * (self.fc * 1e9) / self.c
            h = 5
            W = 20
            for i in range(ue_positions.shape[0]):
                #for j in range(ue_positions.shape[1]):
                    ue_position = ue_positions[i]
                    d3D = BS_position - ue_position
                    distance = np.linalg.norm(d3D) * 1000  # Convert to meter
                    d2D = BS_position[:2] - ue_position[:2]
                    d_2D = np.linalg.norm(d2D) * 1000  # Convert to meter
                    
                    prob_los = self.Pr_LOS_RMa(d_2D, h_UT)
                    
                    if self.environment == 'Suburban':
                        if prob_los >= 0.5:
                            PL_RMa_NLOS = 0
                            PL_1 = 20 * math.log10(40 * math.pi * distance * self.fc / 3) + \
                                   min(0.03 * math.pow(h, 1.72), 10) * math.log10(distance) - \
                                   min(0.044 * math.pow(h, 1.72), 14.77) + 0.002 * math.log10(h) * distance
                            PL_2 = PL_1 + 40 * math.log10(distance / d_BP)
        
                            if 10 < d_2D and d_2D <= d_BP:
                                PL_RMa_LOS = PL_1 + UEs_Shadowing_per_BS_from_Gaussian_field[bs, i]
                            elif d_BP < d_2D and d_2D <= 10 * 1000:
                                PL_RMa_LOS = PL_2 + UEs_Shadowing_per_BS_from_Gaussian_field[bs, i]
                        else:
                            PL_RMa_LOS = 0
                            PL_RMa_NLOS_2 = 161.04 - 7.1 * math.log10(40 * W) + 7.5 * math.log10(h) - \
                                            (24.37 - 3.7 * (h / h_BS) * (h / h_BS)) * math.log10(h_BS) + \
                                            (43.42 - 3.1 * math.log10(h_BS)) * (math.log10(distance) - 3) + \
                                            20 * math.log10(self.fc) - (3.2 * math.pow(math.log10(11.75 * h_UT), 2) - 4.97)
                            if 10 < d_2D and d_2D <= 5 * 1000:
                                PL_RMa_NLOS = PL_RMa_NLOS_2 + UEs_Shadowing_per_BS_from_Gaussian_field[bs, i]
        
                        PL_Sha_LOS[i] = PL_RMa_LOS
                        PL_Sha_NLOS[i] = PL_RMa_NLOS
            PL_Shad_LOS[bs].extend(PL_Sha_LOS)
            PL_Shad_NLOS[bs].extend(PL_Sha_NLOS)
        aa = np.array([PL_Shad_LOS])
        bb = np.array([PL_Shad_NLOS])
    
        return aa, bb # at the end of the day it must be as much as number of UEs 

##################################################################################################
    def elevation_angel_calculator(self, P_sat, P_gs):
        sat_pos_ecef = P_sat
        obs_pos_ecef = P_gs
        # Calculate vector from observer to satellite
        r_obs_sat = sat_pos_ecef - obs_pos_ecef
        # Convert vector to local-level coordinates:
         # Define WGS84 ellipsoid constants
        a = 6378137.0  # semimajor axis (m)
        b = 6356752.314245  # semiminor axis (m)
        f = (a - b) / a  # flattening
        e_sq = f * (2 - f)  # eccentricity squared
        # Calculate magnitude of position vector
        r = np.linalg.norm(obs_pos_ecef)
        # Calculate latitude
        obs_lat = math.atan2(obs_pos_ecef[2], math.sqrt(obs_pos_ecef[0]**2 + obs_pos_ecef[1]**2))
        # Calculate longitude
        obs_lon = math.atan2(obs_pos_ecef[1], obs_pos_ecef[0])
        # Calculate altitude
        N = a / math.sqrt(1 - e_sq * math.sin(obs_lat)**2)
        alt = r - N
        # Convert latitude and longitude to degrees
        obs_lat = math.radians(math.degrees(obs_lat)+0.18)
        obs_lon = math.radians(math.degrees(obs_lon))
        #obs_lat = math.radians(53.105750)  # observer's latitude in radians
        #obs_lon = math.radians(8.859860)  # observer's longitude in radians
        obs_n = -np.sin(obs_lat) * np.cos(obs_lon) * r_obs_sat[0] - np.sin(obs_lat) * np.sin(obs_lon) * r_obs_sat[1] + np.cos(obs_lat) * r_obs_sat[2]
        obs_e = -np.sin(obs_lon) * r_obs_sat[0] + np.cos(obs_lon) * r_obs_sat[1]
        obs_u = np.cos(obs_lat) * np.cos(obs_lon) * r_obs_sat[0] + np.cos(obs_lat) * np.sin(obs_lon) * r_obs_sat[1] + np.sin(obs_lat) * r_obs_sat[2]
        # Calculate elevation angle in degrees
        elevation_angle = math.degrees(math.atan2(obs_u, math.sqrt(obs_e**2 + obs_n**2)))
        if elevation_angle < 10:
            elevation_angle = 10
        return elevation_angle
    
    def elevation_angle_Sat_BSs(self, base_station_positions, sat_position):
        elevation_angles = []
        for bs_position in base_station_positions:
            elevation = self.elevation_angel_calculator(sat_position, bs_position)
            elevation_angles.append(elevation)
        return elevation_angles
    """        
    def Sat_to_Bs_Interference(self, distances, elevation_angels, A_z, channel, Sat_EIRP):
        sat_BS_interference = []
        for d in range(len(distances)):
            elev_angle_radian = np.deg2rad(elevation_angels[d])
            PL_At = 10*np.log10((10**(A_z/10))/np.sin(elev_angle_radian))
            p_rx = (Sat_EIRP+5) + 10*np.log10(channel[d]) - (32.45+20*np.log10(self.fc) + 20*np.log10(distances[d]*10**3) + PL_At)
            sat_BS_interference.append(p_rx)
        return sat_BS_interference
    """
    def Sat_to_Bs_Interference(self, distances, elevation_angels, A_z, Sat_EIRP):
        sat_BS_interference = []
        for d in range(len(distances)):
            elev_angle_radian = np.deg2rad(elevation_angels[d])
            PL_At = 10*np.log10((10**(A_z/10))/np.sin(elev_angle_radian))
            p_rx = (Sat_EIRP+5) - (32.45+20*np.log10(self.fc) + 20*np.log10(distances[d]*10**3) + PL_At)
            sat_BS_interference.append(p_rx)
        return sat_BS_interference
    
    def Satellite_to_UE_interefernce(self, Sat_Bs_interefrence, elevation_angels, UE_per_BS_array, number_points):
        aa2 = UE_per_BS_array.reshape(3,number_points)
        Sat_Bs_interefrence = np.array([Sat_Bs_interefrence])
        intak = Sat_Bs_interefrence.reshape(3)
        sat_UEs_interference = []
        for i in range(len(intak)):
            shadowing_los, shadowing_nlos, cl_nlos = self.SF_LOS_calc(elevation_angels[i], 'LOS', 'SBand', number_points)
            ue_int = np.full_like(aa2[i,:].shape, intak[i]) + shadowing_los
            sat_UEs_interference.append(ue_int)
        return sat_UEs_interference
    
    def BaseStation_to_GroundStation_Interfeerence(self, BS_position, GS_position, h_BS, h_GS):
        # Here we consider that they follow for S band the same as Pathloss upto 7GHz and for Ka band the follow the formula from the paper 
        # Path Loss Prediction Model for 800 MHz to 37 GHz in NLOS Microcell Environment Koshiro Kitao͊, Tetsuro Imai͊, Ngochao Tran͊, Nobutaka Omaki͊, Yukihiko Okumura͊,
        # Minoru Inomatai, Motoharu Sasakii, and Wataru Yamadai Research Laboratories, NTT DOCOMO, INC. 3-6 Hikari-no-oka, Yokosuka-shi, Kanagawa 239-8536, Japan NTT Access Network Service Systems Laboratories, NTT Corporation
        # At case of Ka band communication defenitely it will be in NLOS and the reason is that this freq band is used for microwave direct to direct communication between Base Stations 
        if self.fc < 10:
            pathloss_los, pathloss_nlos = self.pathloss_calculator_up_to_7GHz(BS_position, GS_position, h_BS, h_GS)
            if pathloss_los == 0:
                loss = 10*np.log10(np.abs(np.random.rayleigh(1,1))**2) - pathloss_nlos
            elif pathloss_nlos == 0:
                loss = 10*np.log10(np.abs(rice.rvs(1,1))**2) - pathloss_los
        else:
            distance = np.linalg.norm(BS_position - GS_position)
            pathloss_NLOS = 36.7 * np.log10(distance*1000) + 20.5 * np.log10(self.fc) + 21.4
            loss = 10*np.log10(np.abs(np.random.rayleigh(1,1))**2) - pathloss_NLOS
        return loss
    
    def convert_to_nearest(self, elevation_angle):
        number = elevation_angle
        # Calculate the nearest smaller and larger numbers
        smaller = int(number) // 10 * 10
        larger = (int(number) // 10 + 1) * 10
    
        # Determine which one is closer
        if abs(number - smaller) <= abs(number - larger):
            return smaller
        else:
            return larger
    
    def SF_LOS_calc(self, elevation_angle, scenario, freq_band, num_points):
        #Define data for Shadowing Faading calculation for suburban scenario, all y values are in dB
        x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]).reshape((-1,))
        
        if freq_band == 'KaBand':
            y_los = np.array([1.9, 1.6, 1.9, 2.3, 2.7, 3.1, 3, 3.6, 0.4])
            y_nlos = np.array([10.7, 10.0, 11.2, 11.6, 11.8, 10.8, 10.8, 10.8, 10.8])
            cl_nlos = np.array([29.5, 24.6, 21.9, 20.0, 18.7, 17.8, 17.2, 16.9, 16.8])
            
        elif freq_band == 'SBand':
             y_los = np.array([1.79, 1.14, 1.14, 0.92, 1.42, 1.56, 0.85, 0.72, 0.72])
             y_nlos = np.array([8.93, 9.08, 8.78, 10.25, 10.56, 10.74, 10.17, 11.52, 11.52])
             cl_nlos = np.array([19.52, 18.17, 18.42, 18.28, 18.63, 17.68, 16.50, 16.30, 16.30])
             
        x_value = self.convert_to_nearest(elevation_angle)
        
        if scenario == 'LOS':
            Sigma_SF_los = np.interp(x_value, x, y_los)
            SF_los = np.random.normal(loc=0, scale=Sigma_SF_los, size=num_points)
            if np.isnan(SF_los.all()):
                SF_los = 0
            SF_nlos = 0
            CL = 0
        elif scenario == 'NLOS':
            SF_los = 0
            Sigma_SF_nlos = np.interp(x_value, x, y_nlos)
            SF_nlos = np.random.normal(loc=0, scale=Sigma_SF_nlos, size=num_points)
            if np.isnan(SF_nlos.all()):
                SF_nlos = 0
            CL = np.interp(x_value, x, cl_nlos)
        return SF_los, SF_nlos, CL
    
    
    
    def doppler_Sat_UE(self, vel_Sat_UEs, alpha_n):
        doppler_Sat_UEs = np.zeros((vel_Sat_UEs.shape[0], alpha_n.shape[0]))
        for i in range(vel_Sat_UEs.shape[0]):
            doppler_Sat_UEs[i,:] = 2*np.pi*self.fc*1e9*vel_Sat_UEs[i]*np.cos(alpha_n)/(3e8)
        return doppler_Sat_UEs

    
    def small_scale_Jake_fading(self, t, M,alpha,phi_array, doppler, aa):
        channel_rayleigh_array_UEs_per_BSs = np.zeros(aa.shape)
        for iak in range(channel_rayleigh_array_UEs_per_BSs.shape[0]):
            for jak in range(channel_rayleigh_array_UEs_per_BSs.shape[1]):
                phi = phi_array[:,iak,jak]
                channel_rayleigh = np.abs(np.sum([(1/np.sqrt(M))*np.exp(1j*doppler[i]*t*np.cos((alpha[i])+2*np.pi)/(M))*np.exp(1j*phi[i]) for i in range(1,M)]))
                channel_rayleigh_array_UEs_per_BSs[iak, jak] = channel_rayleigh
        return channel_rayleigh_array_UEs_per_BSs
    
    def Sat_UE_small_scale_Jakes(self, t, M , alpha , phi_array, doppler_Sat_UEs, aa):
        Sat_UEs_Jakes = np.zeros(aa.shape)
        for iak in range(Sat_UEs_Jakes.shape[0]):
            doppler = doppler_Sat_UEs[iak, :]
            for jak in range(Sat_UEs_Jakes.shape[1]):
                phi = phi_array[:,iak,jak]
                channel_rayleigh = np.abs(np.sum([(1/np.sqrt(M))*np.exp(1j*doppler[i]*t*np.cos((alpha[i])+2*np.pi)/(M))*np.exp(1j*phi[i]) for i in range(1,M)]))
                Sat_UEs_Jakes[iak, jak] = channel_rayleigh
        return Sat_UEs_Jakes
    
    def rician_fading_accurate(self, num_samples, velocity, fs, K, theta_rad):
        N = 256
        t = np.arange(num_samples) / fs
        fd = (velocity / self.c) * self.fc
        omega_m = 2 * np.pi * fd
        
        # Initialize Z to zero
        Z = np.zeros(num_samples, dtype=np.complex128)
        
        for n in range(1, N + 1):
            theta_n = 2 * np.pi * np.random.rand() - np.pi  # theta_{n, k}
            phi_n = 2 * np.pi * np.random.rand() - np.pi    # phi_{n, k}
            Z += np.exp(1j * omega_m * t * np.cos((2 * np.pi * n + theta_n) / N)) * np.exp(1j * phi_n)
        
        Z *= np.sqrt(1 / ((N+1) * (1 + K)))
        # LOS component
        theta_0 = theta_rad
        phi_0 = 2 * np.pi * np.random.rand() - np.pi       # phi_{0, k}
        Z_LOS = np.sqrt(K / (1 + K)) * np.exp(1j * (omega_m * t * np.cos(theta_0) + phi_0))
        fading_process = Z + Z_LOS
        
        return fading_process

    def nakagami_m_based_Gamma(self, m, Omega):
        shape = 2 * m
        scale = Omega / m
        Gamma_samples = np.random.gamma(shape, scale, self.num_samples_nakagami)
        nakagami_sample = np.sqrt(Gamma_samples)
        return nakagami_sample

    def Rank_matching(self, rayleigh_seq, nakagami_seq):
        # Rank matching
        rayleigh_sorted_indices = np.argsort(rayleigh_seq)
        nakagami_sorted_indices = np.argsort(nakagami_seq)
        nakagami_sorted = nakagami_seq[nakagami_sorted_indices]
        nakagami_matched = np.empty_like(nakagami_sorted)
        nakagami_matched[rayleigh_sorted_indices] = nakagami_sorted
        return nakagami_matched
        