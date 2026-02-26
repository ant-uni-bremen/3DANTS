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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy.special as sc
from scipy.special import j1, jv, hyp1f1, iv

class Rx_power():
    
    np.random.seed(42)
    
    
    def ShadowedRicianRandGen(self, environemnt, N):
        # Three different shadowed-Rician fading models are taken into consideration: frequent heavy shadowing (FHS) {b = 0.063, m = 0.739, Ω = 8.97×104}, average shadowing (AS) {b = 0.126, m = 10.1, Ω = 0.835}, and infrequent light shadowing (ILS) {b = 0.158, m = 19.4, Ω = 1.29}
        if environemnt == 'Heavy':
            b0 = 0.063; m = 0.739; omegaa = 8.97e4
        elif environemnt == 'Average':
            b0 = 0.126; m = 10.1;  omegaa= 0.835
        elif environemnt == 'Light':
            b0 = 0.158; m = 20; omegaa = 1.3
        x = np.logspace(-1, 1, N)
        alphaa = (((2 * b0 * m) / (2 * b0 * m + omegaa))**m) / (2 * b0)
        bettaa = 1 / (2 * b0)
        gammaa = omegaa / (2 * b0 * (2 * b0 * m + omegaa))
        
        f = alphaa * np.exp(-bettaa * x) * sc.hyp1f1(m, 1, gammaa * x)
        
        # Inverse Transform Sampling
        dx = x[1:] - x[:-1]
        c = np.cumsum(f[:-1] * dx)
        
        # Perform interpolation
        cq = np.random.uniform(0, 1, N)
        cq_sorted = np.sort(cq)
        xq = np.interp(cq_sorted, c, x[:-1])
        return xq
    
    def Rician_Gen(self, K , omega, N):
        x = np.logspace(-1, 1, N)
        f = ((2*(1 + K)*np.exp(-K)*x)/omega)*np.exp((-1*(1+K)*x**2)/omega)*sc.iv(0, x*2*np.sqrt(K)*np.sqrt((1+K)/omega))
        # Inverse Transform Sampling
        dx = x[1:] - x[:-1]
        c = np.cumsum(f[:-1] * dx)
        # Perform interpolation
        cq = np.random.uniform(0, 1, N)
        cq_sorted = np.sort(cq)
        xq_rice = np.interp(cq_sorted, c, x[:-1])
        return xq_rice
    
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
    
    def LOS_prob_calc(self, elevation_angle, environment):
        # Define the data for suburban scenario
        x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]).reshape((-1, 1))
        if environment == 'Dense_Urban':
            y = np.array([28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1])
        elif environment == 'Urban':
            y = np.array([24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2])
        elif environment == 'Sub_Urban':
            y = np.array([78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8])

        # Transform the input data to include polynomial terms up to degree 3
        poly = PolynomialFeatures(degree=3)
        x_poly = poly.fit_transform(x)

        # Create a linear regression object
        model = LinearRegression()

        # Train the model on the transformed data
        model.fit(x_poly, y)
        x_new = np.array([[elevation_angle]])
        x_new_poly = poly.transform(x_new)
        y_new = model.predict(x_new_poly)
        return y_new
    
    def SF_LOS_calc(self, elevation_angle, scenario, freq_band):
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
            SF_los = np.random.normal(loc=0, scale=Sigma_SF_los)
            if np.isnan(SF_los):
                SF_los = 0
            SF_nlos = 0
            CL = 0
        elif scenario == 'NLOS':
            SF_los = 0
            Sigma_SF_nlos = np.interp(x_value, x, y_nlos)
            SF_nlos = np.random.normal(loc=0, scale=Sigma_SF_nlos)
            if np.isnan(SF_nlos):
                SF_nlos = 0
            CL = np.interp(x_value, x, cl_nlos)
        return SF_los, SF_nlos, CL
    
    def antenna_gain_calc(self, theta):
        # Constants
        eta = 0.7
        N = 70
        theta_3dB = 5.12
        theta_3dB_rad = np.radians(theta_3dB)

        # Calculate u(theta)
        u_theta = 2.07123 * np.sin(np.radians(theta)) / np.sin(theta_3dB_rad)

        # Calculate G0
        G0 = (eta * N**2 * np.pi**2) / theta_3dB**2
        #G0 = 10**(35/10)

        # Calculate G(theta)
        G_theta = G0 * (j1(u_theta) / (2 * u_theta) + 36 * jv(3, u_theta) / u_theta**3)
        if G_theta < 0 :
            gain_dB = -1
        else:
            # Convert gain to dB using 10*log10
            gain_dB = 10 * np.log10(G_theta)
            # Replace -inf values with a large negative number
            gain_dB = np.where(np.isnan(gain_dB), 0, gain_dB)

        return gain_dB
    
    def calculate_angle_interf(self,GS_pos, sat_pos, interferer_pos):
        x_m = GS_pos[0]; y_m = GS_pos[1]; z_m = GS_pos[2];
        x_s = sat_pos[0]; y_s = sat_pos[1]; z_s = sat_pos[2];
        x_i = interferer_pos[0]; y_i = interferer_pos[1]; z_i = interferer_pos[2];
        # Calculate the vectors
        vector_sm = np.array([x_s - x_m, y_s - y_m, z_s - z_m])
        vector_im = np.array([x_i - x_m, y_i - y_m, z_i - z_m])
        # Calculate the magnitudes of the vectors
        magnitude_sm = np.linalg.norm(vector_sm)
        magnitude_im = np.linalg.norm(vector_im)
        # Calculate the dot product
        dot_product = np.dot(vector_sm, vector_im)
        # Calculate the cosine of the angle
        cosine_angle = dot_product / (magnitude_sm * magnitude_im)
        # Calculate the angle in radians
        angle_rad = np.arccos(cosine_angle)
        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    def FSPl_only(self, freq, distance):
        fspl_solo_dB = 32.45+20*np.log10(freq/1e9) + 20*np.log10(distance*10**3)
        return fspl_solo_dB
    
    def Path_loss(self, freq, distance, elevation_angle, scenario, band):
        fspl_dB = 32.45+20*np.log10(freq/1e9) + 20*np.log10(distance*10**3)
        if scenario == 'LOS':
            y_value_los, y_value_nlos, y_cl = self.SF_LOS_calc(elevation_angle, scenario = 'LOS', freq_band = band)
        elif scenario == 'NLOS':
            y_value_los, y_value_nlos, y_cl = self.SF_LOS_calc(elevation_angle, scenario = 'NLOS', freq_band = band)
        PL_dB = fspl_dB + y_value_los + y_value_nlos + y_cl
        return PL_dB
    
    
    #Calculator of the atmospheric loss based on page 52 of 3gpp 38.811
    def atmospheric_att(self, A_z, elevation_angel):
        #A_z is the Zenith attenuation in dB and is driven from the Fig. 4 of ITU-R P.676-13
        elev_angle_radian = np.deg2rad(elevation_angel)
        PL_At = 10*np.log10((10**(A_z/10))/np.sin(elev_angle_radian))
        return PL_At
    
    
    
    def Noise_power_with_NoiseFigure(self, noise_figure_db, bandwidth_Mhz, temperature_k):
        k = 1.38e-23  # Boltzmann constant in J/K
        noise_factor = 10 ** (noise_figure_db / 10)
        P_thermal = k * temperature_k * (bandwidth_Mhz * 1000000)
        P_noise = 10*np.log10(noise_factor * P_thermal)
        #P_noise_dbm = 10 * np.log10(P_noise / 1e-3)  # converting to dBm
        return P_noise
    
    def small_scale_fading(self, t, M,alpha,phi, doppler):

        channel_rayleigh = np.abs(np.sum([(1/np.sqrt(M))*np.exp(1j*doppler[i]*t*np.cos((alpha[i])+2*np.pi)/(M))*np.exp(1j*phi[i]) for i in range(1,M)]))
        return channel_rayleigh

    def HAP_interferer(self, distance, antenna_gain, channel_gain, elevation_angle, freq):
        if antenna_gain >= 0:
            scenario = 'LOS'
            loss = antenna_gain+10 + 10*np.log10(channel_gain) - (32.45+20*np.log10(freq/1e9) + 20*np.log10(distance*10**3))
        else:
            scenario = 'NLOS'
            if freq/1e9 < 10:
                freq_band = 'SBand'
            else:
                freq_band = 'KaBand'
            y_value_los, y_value_nlos, y_cl = self.SF_LOS_calc(elevation_angle, scenario, freq_band)
            loss = -1*((32.45+20*np.log10(freq/1e9) + 20*np.log10(distance*10**3))+(y_value_los+y_value_nlos+y_cl))
        return loss 
