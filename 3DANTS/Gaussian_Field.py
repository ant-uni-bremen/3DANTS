#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:54:42 2024

@author: vakilifard
"""
import numpy as np
from numpy import linalg as LA
import gstools as gs
from scipy.interpolate import griddata

class Guassian_Random_filed_generator:
    """
    # Set up random field parameters
    seed = gs.random.MasterRNG(19970221)
    rng = np.random.RandomState(seed())"""
    def __init__(self):
        # Set up random field parameters
        self.seed = gs.random.MasterRNG(19970221)
        self.rng = np.random.RandomState(self.seed())
    
    def field_generator_2D(self, dimension, variance, len_sacel, center_x, center_y, radius):
        # Generate the covariance of the model
        model = gs.Stable(dim=2, var=8, len_scale=10/1000, alpha = 1)
        srf = gs.SRF(model, seed=20170519)
        # Create a grid centered at the specified point
        n_points = 10000
        theta = np.linspace(0, 2 * np.pi, n_points)
        self.x_grid = center_x + radius * np.cos(theta)
        self.y_grid = center_y + radius * np.sin(theta)
        # Generate the field on the selected area
        field = srf((self.x_grid, self.y_grid))
        return field, srf
    
    def get_field_value_at_point(self, UEs_point, field, center_coordinates, d_correlation):
        # Use linear interpolation to estimate the field value
        UEs_shadowing_per_BSs = np.zeros((UEs_point.shape[0], UEs_point.shape[1]))
        value_at_center_list = []
        for i in range(UEs_point.shape[0]):
            field_of_each_BS = field[:,i]
            center_x, center_y = center_coordinates[i,:]
            for j in range(UEs_point.shape[1]):
                pointak = (UEs_point[i,j,0], UEs_point[i,j,1])
                value_at_pointak = griddata((self.x_grid, self.y_grid), field_of_each_BS, pointak, method='nearest')
                value_at_center = griddata((self.x_grid, self.y_grid), field_of_each_BS, (center_x, center_y), method='nearest')
                distnace_vec = np.array([center_x, center_y]) - np.array(UEs_point[i,j,0], UEs_point[i,j,1])
                X_coeff_between_two_points = (1-np.exp(-LA.norm(distnace_vec)/d_correlation))/(np.sqrt(2)*np.sqrt(1 + np.exp(-LA.norm(distnace_vec)/d_correlation)))
                value_at_points = 10**(X_coeff_between_two_points/3)*(value_at_pointak+value_at_center)
                UEs_shadowing_per_BSs[i,j] = value_at_points
            value_at_center_list.append(value_at_center)
        return UEs_shadowing_per_BSs, value_at_center_list