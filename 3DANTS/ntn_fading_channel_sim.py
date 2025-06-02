#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:21:46 2025

@author: vakilifard
"""

import numpy as np

class FadingSimulation_Non_terrestrial:
    def __init__(self, num_samples, fs, N, h, Doppler_compensate):
        # Simulation parameters
        self.num_samples = num_samples
        self.fs = fs
        self.N = N
        self.h = h  # Satellite height in meters
        
        # Constants
        self.c = 3e8  # Speed of light in m/s
        self.R = 6371e3  # Earth's radius in meters
        self.G = 6.6743e-11
        self.M = 5.9722e24
        if Doppler_compensate == 'Yes':
            self.v_sat = np.sqrt((self.G * self.M) / ((self.R + self.h) ** 2))  # Satellite velocity in m/s
        else:
            self.v_sat = np.sqrt((self.G * self.M) / ((self.R + self.h)))  # Satellite velocity in m/s

    

    def b_0_calc(self, theta):
        return -4.7943e-8 * theta**3 + 5.5784e-6 * theta**2 - 2.1344e-4 * theta + 3.2710e-2

    def m_theta(self, theta):
        return 6.3739e-5 * theta**3 + 5.8533e-4 * theta**2 - 1.5973e-1 * theta + 3.5156

    def Omega_theta(self, theta):
        return 1.4428e-5 * theta**3 - 2.3798e-3 * theta**2 + 0.12702 * theta - 1.4864

    def f_D(self, theta, fc):
        return (self.v_sat / self.c) * (self.R / (self.R + self.h)) * np.cos(theta) * fc

    def rician_fading_accurate(self, b_0, fd):
        t = np.arange(self.num_samples) / self.fs
        omega_m = 2 * np.pi * fd
        
        # Initialize Z to zero
        Z = np.zeros(self.num_samples, dtype=np.complex128)
        
        for n in range(1, self.N + 1):
            theta_n = 2 * np.pi * np.random.rand() - np.pi  # theta_{n, k}
            phi_n = 2 * np.pi * np.random.rand() - np.pi    # phi_{n, k}
            Z += np.exp(1j * omega_m * t * np.cos((2 * np.pi * n + theta_n) / self.N)) * np.exp(1j * phi_n)

        Z *= np.sqrt(1 / (self.N + 1)) * np.sqrt(2 * b_0)
        fading_process = Z
        
        return fading_process



    def nakagami_m_based_Gamma(self, m, Omega):
        shape = 2 * m
        scale = Omega / m
        Gamma_samples = np.random.gamma(shape, scale, self.num_samples)
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

    def run_simulation(self, theta_degrees, fc_array, distance_satellite):
        if not isinstance(theta_degrees, (list, np.ndarray)):
            theta_degrees = [theta_degrees]
        if not isinstance(fc_array, (list, np.ndarray)):
            fc_array = [fc_array]
            
        theta_radians = np.deg2rad(theta_degrees)
        distance_satellite = distance_satellite * 1000
        for fc in fc_array:
            for theta_deg, theta_rad in zip(theta_degrees, theta_radians):
                fd = self.f_D(theta_rad, fc)
                b_0 = self.b_0_calc(theta_deg)
                # Generate Rayleigh fading sequence which is temporally correlated
                rayleigh_fading_samples = self.rician_fading_accurate(b_0, fd)
                
                m = self.m_theta(theta_deg)
                if m < 0:
                    m = 0.835
                    
                Omega = self.Omega_theta(theta_deg)
                if Omega < 0:
                    Omega = 0.000897
                
                # Generate uncorrelated Nakagami-m samples
                nakagami_sequence = self.nakagami_m_based_Gamma(m, Omega)
                
                # Generate temporally correlated Rayleigh time series with lower Doppler shift
                lower_doppler_rayleigh_fading = self.rician_fading_accurate(b_0, fd / 100)
                
                # Rank matching
                nakagami_correlated_ranked_matched = self.Rank_matching(np.abs(lower_doppler_rayleigh_fading), nakagami_sequence)
                
                # Combine to form the shadowed Rician channel
                Shadowed_rician_channel = rayleigh_fading_samples + nakagami_correlated_ranked_matched * np.exp(-1j * 2 * np.pi * ((distance_satellite * fc)/self.c))
        return Shadowed_rician_channel
    
    
    def channel_with_UE_movement(self, theta_degrees, fc_array, UE_speed, UE_moving_direction,distance_satellite):
        if not isinstance(theta_degrees, (list, np.ndarray)):
            theta_degrees = [theta_degrees]
        if not isinstance(fc_array, (list, np.ndarray)):
            fc_array = [fc_array]
            
        theta_radians = np.deg2rad(theta_degrees)
        distance_satellite = distance_satellite * 1000
        for fc in fc_array:
            for theta_deg, theta_rad in zip(theta_degrees, theta_radians):
                fd = self.f_D(theta_rad, fc)
                fd_due_to_UE_movement = (UE_speed/self.c) * (fc+fd)*np.cos(UE_moving_direction)
                # Generate Rayleigh fading sequence which is temporally correlated
                b_0 = self.b_0_calc(theta_deg)
                rayleigh_fading_samples = self.rician_fading_accurate(b_0, fd_due_to_UE_movement)
                
                m = self.m_theta(theta_deg)
                if m < 0:
                    m = 0.835
                    
                Omega = self.Omega_theta(theta_deg)
                if Omega < 0:
                    Omega = 0.000897
                
                # Generate uncorrelated Nakagami-m samples
                nakagami_sequence = self.nakagami_m_based_Gamma(m, Omega)
                
                # Generate temporally correlated Rayleigh time series with lower Doppler shift
                lower_doppler_rayleigh_fading = self.rician_fading_accurate(b_0, fd_due_to_UE_movement / 100)
                
                # Rank matching
                nakagami_correlated_ranked_matched = self.Rank_matching(np.abs(lower_doppler_rayleigh_fading), nakagami_sequence)
                
                # Combine to form the shadowed Rician channel
                Shadowed_rician_channel = rayleigh_fading_samples + nakagami_correlated_ranked_matched * np.exp(-1j * 2 * np.pi * ((distance_satellite * fc)/self.c))
        return Shadowed_rician_channel
                
