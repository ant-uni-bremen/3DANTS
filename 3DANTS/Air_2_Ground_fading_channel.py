#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 12:59:29 2025

@author: vakilifard
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0  # Import Bessel function
import math

class Air_Fading_channel:
    def __init__(self, num_samples_nakagami, num_samples_rician, fs, N):
        # Simulation parameters
        self.num_samples_nakagami = num_samples_nakagami
        self.num_samples_rician = num_samples_rician
        self.num_segments = int(self.num_samples_nakagami/self.num_samples_rician)
        self.fs = fs
        self.N = N        
        

    def rician_fading_accurate(self, num_samples, fd, fs, K, theta_rad):
        t = np.arange(num_samples) / self.fs
        omega_m = 2 * np.pi * fd
        
        # Initialize Z to zero
        Z = np.zeros(num_samples, dtype=np.complex128)
        
        for n in range(1, self.N + 1):
            theta_n = 2 * np.pi * np.random.rand() - np.pi  # theta_{n, k}
            phi_n = 2 * np.pi * np.random.rand() - np.pi    # phi_{n, k}
            Z += np.exp(1j * omega_m * t * np.cos((2 * np.pi * n + theta_n) / self.N)) * np.exp(1j * phi_n)
        
        Z *= np.sqrt(1 / ((self.N+1) * (1 + K)))
        # LOS component
        theta_0 = theta_rad
        phi_0 = 2 * np.pi * np.random.rand() - np.pi       # phi_{0, k}
        Z_LOS = np.sqrt(K / (1 + K)) * np.exp(1j * (omega_m * t * np.cos(theta_0) + phi_0))
        fading_process = Z + Z_LOS
        
        return fading_process

    def b_0_calc(self, theta):
        return -4.7943e-8 * theta**3 + 5.5784e-6 * theta**2 - 2.1344e-4 * theta + 3.2710e-2

    def m_theta(self, theta):
        return 6.3739e-5 * theta**3 + 5.8533e-4 * theta**2 - 1.5973e-1 * theta + 3.5156

    def Omega_theta(self, theta):
        return 1.4428e-5 * theta**3 - 2.3798e-3 * theta**2 + 0.12702 * theta - 1.4864



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

    def run_simulation(self, theta_degrees, velocity, fc_array, distance_satellite):
        if not isinstance(theta_degrees, (list, np.ndarray)):
            theta_degrees = [theta_degrees]
        if not isinstance(fc_array, (list, np.ndarray)):
            fc_array = [fc_array]
            
        theta_radians = np.deg2rad(theta_degrees)
        distance_satellite = distance_satellite * 1000
        for fc in fc_array:
            for theta_deg, theta_rad in zip(theta_degrees, theta_radians):
                fd = ((np.abs(velocity) * 1000)/(3.0e8)) * fc
                
                # Update fs if necessary
                if self.fs < 2 * fd:
                    self.fs = np.ceil(2 * fd)
                    
                b_0 = self.b_0_calc(theta_deg)
                m = self.m_theta(theta_deg)
                Omega = self.Omega_theta(theta_deg)
                K = Omega/b_0/2
                # Generate three Rician fading segments and concatenate
                rician_segments = []
                for _ in range(self.num_segments):
                    segment = self.rician_fading_accurate(self.num_samples_rician, fd, self.fs, K, theta_rad)
                    rician_segments.append(segment)
                rician_fading_samples = np.concatenate(rician_segments)
                
                # Generate uncorrelated Nakagami-m samples
                nakagami_sequence = self.nakagami_m_based_Gamma(m, Omega)
                
                # Generate temporally correlated rayleigh time series with lower Doppler shift
                lower_doppler_rayleigh_fading = self.rician_fading_accurate(self.num_samples_nakagami, fd/100, self.fs, 0, theta_rad)
                
                # Rank matching
                nakagami_correlated_ranked_matched = self.Rank_matching(lower_doppler_rayleigh_fading, nakagami_sequence)
                
                # Combine to form the shadowed Rician channel
                Shadowed_rician_channel = rician_fading_samples + nakagami_correlated_ranked_matched * np.exp(-1j * 2 * np.pi * (distance_satellite * 1000 * fc) /3e8)
        return Shadowed_rician_channel
                
