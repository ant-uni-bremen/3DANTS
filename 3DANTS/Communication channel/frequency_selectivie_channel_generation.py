#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 19:34:16 2025

@author: vakilifard
"""

import numpy as np
# frequency_selective_fading.py
from ntn_fading_channel_sim import FadingSimulation_Non_terrestrial

class FrequencySelectiveFadingSimulation(FadingSimulation_Non_terrestrial):
    def __init__(self, num_samples, fs, N, h, num_subcarriers, delay_spread, num_taps,
                 doppler_mode='full', v_residual_mps=1.5, Tc_target_s=0.034,
                 fc_ref=2.5e9, Doppler_compensate=None):
        super().__init__(num_samples, fs, N, h,
                         doppler_mode=doppler_mode,
                         v_residual_mps=v_residual_mps,
                         Tc_target_s=Tc_target_s,
                         fc_ref=fc_ref,
                         Doppler_compensate=Doppler_compensate)
        
        self.num_subcarriers = num_subcarriers
        self.delay_spread = delay_spread
        self.num_taps = num_taps
        
    def generate_frequency_correlation_matrix(self, subcarrier_spacing):
        # Generate delays for taps
        delays = np.linspace(0, (self.num_taps - 1) / subcarrier_spacing, self.num_taps)
        
        # Generate exponential PDP
        pdp = np.exp(-delays / self.delay_spread)
        pdp = pdp / np.sum(pdp)
        
        # Generate frequency domain correlation matrix
        freq_correlation = np.zeros((self.num_subcarriers, self.num_subcarriers), dtype=complex)
        
        for i in range(self.num_subcarriers):
            for j in range(self.num_subcarriers):
                delta_f = (i - j) * subcarrier_spacing
                freq_correlation[i, j] = np.sum(pdp * np.exp(-1j * 2 * np.pi * delta_f * delays))
        
        return freq_correlation
    
    def apply_frequency_correlation(self, samples, freq_correlation):
        """
        Apply frequency correlation to a set of samples using Cholesky decomposition
        """
        # Generate complex Gaussian samples
        z = np.random.normal(0, 1, self.num_subcarriers) + \
            1j * np.random.normal(0, 1, self.num_subcarriers)
        
        # Apply frequency correlation through Cholesky decomposition
        L = np.linalg.cholesky(freq_correlation)
        return np.dot(L, z) * samples
    
    def run_frequency_selective_simulation(self, theta_degrees, fc_array, distance_satellite, subcarrier_spacing):
        if not isinstance(theta_degrees, (list, np.ndarray)):
            theta_degrees = [theta_degrees]
        if not isinstance(fc_array, (list, np.ndarray)):
            fc_array = [fc_array]
            
        theta_radians = np.deg2rad(theta_degrees)
        distance_satellite = distance_satellite * 1000
        
        # Generate frequency correlation matrix
        freq_correlation = self.generate_frequency_correlation_matrix(subcarrier_spacing)
        
        # Initialize the channel matrix
        channel_matrix = np.zeros((self.num_samples, self.num_subcarriers), dtype=complex)
        
        for fc in fc_array:
            for theta_deg, theta_rad in zip(theta_degrees, theta_radians):
                fd = self.f_D(theta_rad, fc)
                
                # Generate base time-correlated Rayleigh process
                b_0 = self.b_0_calc(theta_deg)
                time_correlated_rayleigh = self.rician_fading_accurate(b_0, fd)
                
                # Generate Nakagami-m parameters
                m = max(0.835, self.m_theta(theta_deg))
                Omega = max(0.000897, self.Omega_theta(theta_deg))
                
                # Generate time-correlated Nakagami-m component
                nakagami_sequence = self.nakagami_m_based_Gamma(m, Omega)
                lower_doppler_rayleigh = self.rician_fading_accurate(b_0, fd / 100)
                nakagami_correlated = self.Rank_matching(
                    np.abs(lower_doppler_rayleigh), 
                    nakagami_sequence
                )
                
                # Apply frequency correlation for each time instance
                for t in range(self.num_samples):
                    # Apply frequency correlation to Rayleigh component
                    rayleigh_freq_correlated = self.apply_frequency_correlation(
                        time_correlated_rayleigh[t],
                        freq_correlation
                    )
                    
                    # Apply frequency correlation to Nakagami component
                    nakagami_freq_correlated = self.apply_frequency_correlation(
                        nakagami_correlated[t],
                        freq_correlation
                    )
                    
                    # Combine components
                    channel_matrix[t, :] = (rayleigh_freq_correlated + 
                                          nakagami_freq_correlated * 
                                          np.exp(-1j * 2 * np.pi * (distance_satellite * fc) / self.c))
        
        return channel_matrix
