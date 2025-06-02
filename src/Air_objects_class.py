#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:53:01 2023

@author: vakilifard
"""
import numpy as np
import math
from scipy.stats import ncx2
from scipy.stats import rice
import scipy.special as sc

class Air:
    np.random.seed(42)
    
    def __init__(self, environment, fc):
        self.fc = fc # it is in Hz not in GHz
        self.c = 3e8
        self.environment = environment
        if self.environment == 'Suburban':
            self.alpha = 0.1
            self.betta = 750
            self.gamma = 8
        elif self.environment == 'Urban':
            self.alpha = 0.3
            self.betta = 500
            self.gamma = 15
        elif self.environment == 'DenseUrban':
            self.alpha = 0.5
            self.betta = 300
            self.gamma = 20
        elif self.environment == 'HighriseUrban':
            self.alpha = 0.5
            self.betta = 300
            self.gamma = 50
            
        

    def LoS_calculator(self, elevation_angle):
        # Based on the paper of Optimal LAP Altitude for Maximum Coverage: Akram Al-Hourani, Student Member, IEEE, Sithamparanathan Kandeepan, Senior Member, IEEE, and Simon Lardner page 2 and 3
        
        C_a =[
        [9.34E-01, 2.30E-01, -2.25E-03, 1.86E-05],
        [1.97E-02, 2.44E-03, 6.58E-06, None],
        [-1.24E-04, -3.34E-06, None, None],
        [2.73E-07, None, None, None] ]
        
        C_b = [[1.17e+00, -7.56e-02, 1.98e-03, -1.78e-05],
               [-5.79e-03, 1.81e-04, -1.65e-06, None],
               [1.73e-05, -2.02e-07, None, None],
               [-2.00e-08, None, None, None]]
        a = 0.0
        b = 0.0
        for j in range(4):
            for i in range(4 - j):
                if C_a[j][i] is not None:
                    a += C_a[j][i] * (self.alpha * self.betta) ** i * self.gamma ** j
                if C_b[j][i] is not None:
                    b += C_b[j][i] * (self.alpha * self.betta) ** i * self.gamma ** j
        Prob_LoS = 1/(1 + a*np.exp(-b*(np.deg2rad(elevation_angle)-a)))
        return Prob_LoS
    
    def pathloss_calculator_up_to_7GHz(self, GS_position, uav_position, h_BS, h_UT):
        # This pathloss is calculated based on the 3GPP TR 38.901 page 27 which is valid only if the altitude of the object is up to 35 meter. It is reported for RMa (Rural Macro), UMa (Urban Macro) and UMi (Urban Micro) scenario
        # Based on 3gpp claim fc must be in Hz, h_Bs and h_UT and all other parametrs related to distnace be in meter
        
        d_BP = 2*np.pi*h_BS*h_UT*(self.fc*1e9)/self.c
        d3D = GS_position-uav_position
        distance = np.linalg.norm(d3D)*1000 #convert it to meter
        d2D = GS_position[:2] - uav_position[:2]
        d_2D = np.linalg.norm(d2D)
        if self.environment == 'Suburban':
            h = 5
            W = 20
            PL_RMa_LOS = 0.0
            PL_RMa_NLOS = 0.0
            PL_1 = 20*math.log10(40*math.pi*distance*self.fc/3) +  min(0.03*math.pow(h, 1.72) , 10)*math.log10(distance) - min(0.044*math.pow(h, 1.72) , 14.77) + 0.002*math.log10(h) *distance
            PL_2 = PL_1 + 40*math.log10(distance/d_BP)
            
            if  10 < d_2D and d_2D <= d_BP:
                PL_RMa_LOS = PL_1 + 4 
            
            if  d_BP < d_2D and d_2D <= 10*1000:
                PL_RMa_LOS = PL_2 + 6
                              
            PL_RMa_NLOS_2 = 161.04 - 7.1*math.log10(40*W) + 7.5*math.log10(h) -(24.37 - 3.7*(h/h_BS)*(h/h_BS))*math.log10(h_BS) + (43.42 - 3.1 * math.log10(h_BS))*(math.log10(distance)-3) + 20*math.log10(self.fc) - (3.2*math.pow(math.log10(11.75*h_UT), 2) - 4.97)
            if 10 < d_2D and d_2D <= 5*1000:
                PL_RMa_NLOS = max(PL_RMa_LOS, PL_RMa_NLOS_2) + 8
            return PL_RMa_LOS, PL_RMa_NLOS
        
    def general_pathloss_calculator(self,GS_position, uav_position):
        # This fumction calculate the pathloss for any frequency and any altitude based on the paper Optimal LAP Altitude for Maximum Coverage Akram Al-Hourani, Student Member, IEEE, Sithamparanathan Kandeepan, Senior Member, IEEE, and, Simon Lardner
        PL_LoS = 0; PL_NLoS = 0;
        # Reshape GS_position to a 1x3 array
        GS_position = GS_position.reshape(1, 3)
        d3D = GS_position-uav_position
        distance = np.linalg.norm(d3D)
        if self.environment == 'Suburban':
            etta_LoS, etta_NLoS = np.array([0.1, 21])            
        elif self.environment == 'Urban':
            etta_LoS, etta_NLoS = np.array([1, 20])
        elif self.environment == 'Denseurban':
            etta_LoS, etta_NLoS = np.array([1.6, 23])
        elif self.environment == 'Highriseurban':
            etta_LoS, etta_NLoS = np.array([2.3, 34])
        PL_LoS = 20*np.log10(distance*1000) + 20*np.log10(self.fc) + 20*np.log10(4*np.pi/self.c) + etta_LoS
        PL_NLoS = 20*np.log10(distance*1000) + 20*np.log10(self.fc) + 20*np.log10(4*np.pi/self.c) + etta_NLoS
        return PL_LoS, PL_NLoS
    
    
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
        
    def Shadow_fading_Air_to_Ground(self, elevation_angle, scenario, freq_band):
        #Define data for Shadowing Faading calculation for suburban scenario, all y values are in dB
        # For now this exactly like the shadow fading of a Non-Terresterial link as specified in 3gpp TR 38.811
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
        
        if scenario == 'LoS':
            Sigma_SF_los = np.interp(x_value, x, y_los)
            SF_los = np.random.normal(loc=0, scale=Sigma_SF_los)
            if np.isnan(SF_los):
                SF_los = 0
            SF_nlos = 0
            CL = 0
        elif scenario == 'NLoS':
            SF_los = 0
            Sigma_SF_nlos = np.interp(x_value, x, y_nlos)
            SF_nlos = 10*np.log10(np.random.normal(loc=0, scale=Sigma_SF_nlos))
            if np.isnan(SF_nlos):
                SF_nlos = 0
            CL = np.interp(x_value, x, cl_nlos)
        return SF_los, SF_nlos, CL
   
    def Rician_factor_calculator(self, elevation_angle, K0, K_pi_half):
        # Based on the paper of Ultra Reliable UAV Communication Using Altitude and Cooperation Diversity Mohammad Mahdi Azari, Fernando Rosas, Kwang-Cheng Chen, and Sofie Pollin
        # page 336, VI. CASE STUDY:APARTICULAR DEPENDENCY FOR α AND K OVER θ
        # convert Ks to linear
        K0 = 10**(K0/10); K_pi_half = 10**(K_pi_half/10)
        a3 = K0; b3 = (2/np.pi)*np.log(K_pi_half/K0)
        K = a3*np.exp(b3*np.deg2rad(elevation_angle))
        #K = np.sqrt(K/(1+K))
        #Based on the Rician channel model introduced in the paper Outage Probability of UAV Communications in the Presence of Interference by Minsu Kim and Jemin Lee, page 3
        #The channel gain is genefrated based on Python self build non central Chi-square random variable generator. The df is the k such that makes the Bessel function of first order to order zero, so it is 2
        #the nc is lambda or shape parameter and is equal to 2*K_Rician which itself is function of elevation angle
        #channel_gain = ncx2.rvs(df=2, nc = 2*np.sqrt(self.K_Rician/(self.K_Rician + 1)), size=1)
        #channel_gain = rice.rvs(np.sqrt(K/(K+1)), size=1)
        x = np.logspace(-1, 1, 100)
        omega = 1
        f = ((2*(1 + K)*np.exp(-K)*x)/omega)*np.exp((-1*(1+K)*x**2)/omega)*sc.iv(0, x*2*np.sqrt(K)*np.sqrt((1+K)/omega))
        # Inverse Transform Sampling
        dx = x[1:] - x[:-1]
        c = np.cumsum(f[:-1] * dx)
        # Perform interpolation
        cq = np.random.uniform(0, 1, 100)
        cq_sorted = np.sort(cq)
        xq_rice = np.interp(cq_sorted, c, x[:-1])
        channel_gain = np.sqrt(K/(1+K))*np.random.choice(a=xq_rice, size=1, replace='True')
        return channel_gain
    
    
    def air2air_K_calculator(self, h1, h2, rho_direct):
        # This function calculates the respected parameter of rho/sigma^2 to generate the Rician RV for an Air 2 Air channel small scale fading based on the paper Investigation of Air-to-Air Channel Characteristics
        #and a UAV Specific Extension to the Rice Model Niklas Goddemeier and Christian Wietfeld, formula 1. we set the K before and get it as an input, 1st calculate the sigma_zero based on the formula 5
        # of the same paper and then calculate the rho and feed to the scipy Rician RV generator:
        a = 212.3; b = -2.221; c = 1.289
        sigma = a*((np.abs(h1-h2))**b) + c
        K = (rho_direct**2)/(2 * sigma**2)
        return K
       
    def air2air_Rician_channel_calculator(self, h1, h2, rho_direct):
        # This function calculates the respected parameter of rho/sigma^2 to generate the Rician RV for an Air 2 Air channel small scale fading based on the paper Investigation of Air-to-Air Channel Characteristics
        #and a UAV Specific Extension to the Rice Model Niklas Goddemeier and Christian Wietfeld, formula 1. we set the K before and get it as an input, 1st calculate the sigma_zero based on the formula 5
        # of the same paper and then calculate the rho and feed to the scipy Rician RV generator:
        a = 212.3; b = -2.221; c = 1.289
        sigma = a*((np.abs(h1-h2))**b) + c
        omega = 1
        K = (rho_direct**2)/(2 * sigma**2)
        #K = np.sqrt(K/(1+K))
        #ssf = rice.rvs(np.sqrt(b/(b+1)), size=1)
        x = np.logspace(-1, 1, 100)
        f = ((2*(1 + K)*np.exp(-K)*x)/omega)*np.exp((-1*(1+K)*x**2)/omega)*sc.iv(0, x*2*np.sqrt(K)*np.sqrt((1+K)/omega))
        # Inverse Transform Sampling
        dx = x[1:] - x[:-1]
        c = np.cumsum(f[:-1] * dx)
        # Perform interpolation
        cq = np.random.uniform(0, 1, 100)
        cq_sorted = np.sort(cq)
        xq_rice = np.interp(cq_sorted, c, x[:-1])
        ssf = np.sqrt(K/(1+K))*np.random.choice(a=xq_rice, size=1, replace='True')
        return ssf
    
    def Air2Air_pathloss_calculator(self, uav1_position, uav2_position):
        # This function calculates the pathloss + shadowing of a uav to uav communication based on the results of the paper An Experimental mmWave Channel Model for UAV-to-UAV Communications
        #Michele Polese, Lorenzo Bertizzolo, Leonardo Bonati Abhimanyu Gosain, Tommaso Melodia. it is reposrted in part 3.2 Comparison of CI and FI Fits
        # Her we use The Close-in free space reference (CI) data fit from Table 1
        d3D = uav1_position - uav2_position
        distance = np.linalg.norm(d3D)
        n_CI = 2.25; sigma_CI = 3.56
        PL_CI = 20*np.log10(4*np.pi*self.fc/self.c) + 10*n_CI*np.log10(distance*1000) + np.random.normal(loc = 0, scale = sigma_CI, size = 1)
        #PL_CI = 20*np.log10(4*np.pi*self.fc/self.c) + 10*n_CI*np.log10(distance*1000)
        return PL_CI
    def Air2Air_pathloss(self, uav1_position, uav2_position):
        # This function calculates the pathloss of a uav to uav communication based on the results of the paper An Experimental mmWave Channel Model for UAV-to-UAV Communications
        #Michele Polese, Lorenzo Bertizzolo, Leonardo Bonati Abhimanyu Gosain, Tommaso Melodia. it is reposrted in part 3.2 Comparison of CI and FI Fits
        # Her we use The Close-in free space reference (CI) data fit from Table 1
        d3D = uav1_position - uav2_position
        distance = np.linalg.norm(d3D)
        n_CI = 2.25
        #PL_CI = 20*np.log10(4*np.pi*self.fc/self.c) + 10*n_CI*np.log10(distance*1000) + np.random.normal(loc = 0, scale = sigma_CI, size = 1)
        PL_CI = 20*np.log10(4*np.pi*self.fc/self.c) + 10*n_CI*np.log10(distance*1000)
        return PL_CI
        
    
    def assign_uavs_to_bs(self, uav_positions, bs_positions):
        num_bs = uav_positions.shape[0];  num_uavs = uav_positions.shape[1]
        num_bs_positions = bs_positions.shape[0]
    
        if num_bs != num_bs_positions:
            raise ValueError("Number of base stations should match the size of bs_positions.")
    
        distances = np.zeros((num_bs, num_uavs))
    
        # Calculate distances between each UAV and each base station
        for i in range(num_bs):
            for j in range(num_uavs):
                distances[i, j] = np.linalg.norm(uav_positions[i,j,:] - bs_positions[i, :])
    
        # Find the index of the closest base station for each UAV
        closest_bs_indices = np.argmin(distances, axis=0)
    
        # Assign each UAV to the closest base station
        assigned_uavs = np.zeros_like(uav_positions)
    
        for i in range(num_uavs):
            assigned_uavs[:, i, :] = bs_positions[closest_bs_indices[i], :]
    
        return assigned_uavs