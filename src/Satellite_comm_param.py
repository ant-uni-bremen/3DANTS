#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:40:22 2023

@author: vakilifard
"""


class Satellite_communication_parameter:
    def parameters(self, freq, band_type, direction):
        self.fc = freq
        if direction == 'DL':
            if band_type == 'S':
                satellite_EIRP_density = 34 # dBW/MHz
                satellite_Tx_max_Gain = 30 #dBi
                satellite_3dB_beamwidth = 4.4127 # degree
                satellite_beam_diameter = 50 # km
                max_Bandwidth_per_beam = 30 # MHz
            elif band_type == 'Ka':
                satellite_EIRP_density = 40 # dBW/MHz
                satellite_Tx_max_Gain = 38.5 #dBi
                satellite_3dB_beamwidth = 1.7647 # degree
                satellite_beam_diameter = 20 # km
                max_Bandwidth_per_beam = 400 # MHz
            return satellite_EIRP_density, satellite_Tx_max_Gain, satellite_3dB_beamwidth, satellite_beam_diameter,  max_Bandwidth_per_beam
        elif direction == 'UL':
            if band_type == 'S':
                G_over_T = 1.1 # dB/K
                max_Bandwidth_per_beam = 30 # MHz
                satellite_Rx_max_gain = 30 #dBi
            elif band_type == 'Ka':
                 G_over_T = 13 # dB/K
                 satellite_Rx_max_gain = 38.5 #dBi
                 max_Bandwidth_per_beam = 400 # MHz
            return G_over_T, max_Bandwidth_per_beam, satellite_Rx_max_gain 